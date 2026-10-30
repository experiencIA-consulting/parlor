# AGENTS.md — contexto para agentes de código

Este archivo registra la migración de Parlor a **arquitectura 100 % cloud (OpenAI / OpenRouter)**, realizada en esta rama (`HP240RG10-dev`). Sirve como dossier de requisitos, plan, cambios y observaciones de depuración para que los futuros agentes no redescubran lo mismo. Escrito en español por convención del mantenedor; los identificadores de código quedan en inglés.

Última actualización: 2026-10-30.

---

## 1. Requisitos que impulsa el mantenedor

1. **Cero modelos LLM locales.** Prohibido descargar llama.cpp, GGUFs, gemma, kokoro-onnx, misaki, etc. Todo el LLM/STT/TTS se consume vía API compatible con OpenAI. El único peso permitido es el clasificador **smart-turn-v3 (~3 MB)** que ya era una dependencia separada.
2. **Proveedores configurables.** Chat conversacional: OpenAI directo (`api.openai.com`) o cualquier endpoint compatible (OpenRouter) vía `OPENAI_BASE_URL` + `OPENAI_MODEL`. STT/TTS siempre por la REST de OpenAI (los endpoints tipo OpenRouter no ofrecen speech).
3. **Clave de OpenAI del mantenedor** en `.env` (ignorada por git; **nunca** hacer echo de su contenido ni pedirla de nuevo — ya está validada contra chat/TTS/STT).
4. **Windows-first.** El entorno de desarrollo es Windows (no macOS/Linux): permisos denegados en `~\.cache\huggingface` (os error 5), `stdout` pipeado a cp1252, ausencia de `bash`/`lsof/pgrep/kill`. Todo debe funcionar sin prerequisitos POSIX.
5. **Conservar el diseño v2 y portarlo.** Bucle de turnos, modos (conversation/translate/listen), timers, delegación de investigación, barge-in, historial server-owned — la semántica del on-device build se mantiene; solo cambia el motor.

## 2. Plan de implementación (mapeo viejo → nuevo)

| Antes (v2 on-device)             | Ahora (v2.1 cloud)                                          |
| -------------------------------- | ------------------------------------------------------------ |
| llama-server + Gemma 4 (audio nativo) | STT OpenAI (`/audio/transcriptions`) → chat streaming OpenAI-compatible |
| Kokoro TTS local                 | TTS OpenAI (`/audio/speech`) por frase; Kokoro queda detrás de `TTS_BACKEND=kokoro` |
| Audio PCM del usuario en el prompt (###TRANSCRIPT:) | Transcripción previa: el servidor llama a STT y el transcript es texto plano enviado al cliente justo tras el STT (sin `###TRANSCRIPT:` en el prompt) |
| Tags en banda (<delegate> etc.)  | Igual que v2: cabeza JSON estructurada (`actions.py`) sobre el endpoint cloud |
| Cache de prefijos de llama-server (priming mid-utterance) | Sin priming: `prime_cache()` es no-op documentado |
| Rotación por `prompt_tokens` real de llama-server | Rotación por `usage.prompt_tokens` del chunk final del stream + estimación (`HISTORY_ROTATE_TOKENS`, default 24000) |

Diagrama objetivo (ver README): browser mic/cam → FastAPI → smart-turn (local, 0.2 s) → STT (cloud) → chat streaming (cloud) → action head (JSON) → TTS (cloud) → browser.

## 3. Cambios realizados (detalle por archivo)

### Nuevos
- **`src/parlor/cloud.py`** — backend cloud completo (interfaz espejo de `llama.py` para no tocar el resto): `chat_blocking` (con `json_schema` → OpenAI structured outputs; añade `additionalProperties: False` requerido por strict-mode, y reintenta sin schema ante HTTP 400), `ChatStream` (streaming SSE con `cancel()` real que cierra el socket; `stream_options.include_usage` para pedir `prompt_tokens` real), `transcribe()` (multipart WAV) y `parse_json` tolerante. `_token_field()` usa `max_completion_tokens` para los modelos razonadores-de-OpenAI (`o*`/`gpt-5`).
- **`benchmarks/check_turns.py`** — clasifica TODOS los WAV de fixtures contra smart-turn localmente (coste cero): usado para reajustar fixtures tras cambiar la voz de TTS.
- **`benchmarks/voice_probe.py`** — sintetiza un texto con cada voz de OpenAI TTS y clasifica: tabla p(complete) por voz. De ahí salió el retune (ver §4). Coste aproximado: unos centavos.

### Reescritos
- **`src/parlor/server.py`** — bucle de turnos STT-first: transcript frame se envía ANTES del LLM; guardas no-words/echo con `speak_only()`; listen-mode transcribe-only sin llamada al LLM (`remember_user_only`); prompts nuevos RESPOND/FLUSH/TRANSLATE (se eliminó el protocolo `###TRANSCRIPT:`); echo-guard server-side (`looks_like_echo`, racha de 8 palabras de la última respuesta); rotación de historial con `HISTORY_ROTATE_TOKENS`; **crítico en Windows**: `sys.stdout/stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)` — ver §4.1.
- **`src/parlor/pipeline.py`** — `merge_wavs()` (concat + cola 0.3 s de silencio), `speak_only()` (respeta orden de frames turn_final→audio_end), `estimate_tokens` solo texto/imagen, swap del parser stream de transcript → respuesta directa + corte `##`. **`run_turn(..., instruction: str | None)`** nuevo parámetro para el guard de instruction-echo (ver §4.4).
- **`src/parlor/tts.py`** — `CloudBackend` (OpenAI TTS) con mapeo de voces: `af_*` → default/`alloy`, `am_*` → `echo`, voces válidas pasan; `TTS_BACKEND=kokoro` conserva MLX/ONNX de la era on-device.
- **`src/parlor/actions.py`** — decisor sobre `cloud.chat_blocking` + schema estricto; cláusulas por modo intactas; fallo ⇒ `NONE` (turno no-op).
- **`src/parlor/reasoner.py`** — razonador cloud con degradación a la clave principal `OPENAI_API_KEY` + modelo `gpt-4o-search-preview` cuando no hay `REASONER_*`.

### Ajustados
- **`benchmarks/fixtures.py`** — retune tras el swap de voces (§4.5): `long_question` ahora se sintetiza con voz **shimmer** (clasifica complete p=0.65), nuevo variant **`long_question_held`** en **alloy** (clasifica HOLD p=0.07); se retiró `long_question_male` (el mapping `am_*`→echo ahora lee completo). Comentarios actualizados con el método.
- **`tests/test_conversation.py`** — test de flush re-apuntado a `long_question_held`; el recuento de "Primed cache" eliminado (el priming es no-op; el propósito real del test es transcripción LIMPIA sobre chunks fusionados).
- **`tests/util.py`** — `collect_turn` ya no permite que `turn_final.transcription` (siempre `None` en protocolo cloud) pise el valor de la frame `transcript` temprana; replica el cliente (§4.3).
- **`tests/conftest.py`** — `PYTHONUTF8=1` en el env del servidor de la suite; `LLAMA_CTX` obsoleto removido; teardown ya no lanza `bash -c lsof/kill` en Windows (generaba basura UTF-16 "Acceso denegado"); no hay child llama que barrer.
- **`tests/test_z_failure.py`**, **`tests/test_llama_startup.py`** — skippidas con motivo "backend llama.cpp sin uso en la edición cloud" (pgrep/kill también son inexistentes en Windows).
- **`tests/test_stream_parser.py`** — unit tests del echo-guard actualizados al texto-only.
- **`pyproject.toml`** — versión 2.1.0; deps: fuera kokoro/mlx/misaki/num2words; dentro `pillow` (marco de cámara de fixtures — se había caído por accidente), `huggingface-hub` para smart-turn; `requires-python = ">=3.12,<3.13"` (uv instala 3.12.x automáticamente).
- **`src/parlor/turn_detector.py`** — todos los env de caché HF (`HF_HOME`, `HF_HUB_CACHE`, `HF_XET_CACHE_DIR`, `HF_HUB_DISABLE_XET`, `HF_HUB_DISABLE_SYMLINKS_WARNING`) se fijan a `.hf-cache/` del proyecto — en Windows el usuario no pudo escribir `~\.cache\huggingface` (os error 5).
- **`src/parlor/web/index.html`** — pill "On-device" → `{{model}}`.
- **`src/parlor/web/static/app.js`** — `turn_final` con `transcription: null` ya no rellena el bubble del usuario con `…` después de una transcript buena (§4.3); se evitó pisar la transcripción temprana.
- **`.env.example` / `README.md` / `docs/configuration.md`** — reflejan la arquitectura cloud: `OPENAI_*`, `OPENAI_STT_LANGUAGE`, `OPENAI_TTS_VOICE`, `TTS_BACKEND`, `HISTORY_ROTATE_TOKENS`; requisitos cloud documentados; coste estimado ~1–2 ct/min; sección Testing corregida (el reasoner es el único mock; chat/STT/TTS usan la clave real).
- **`llama.py` se mantiene en el árbol** (sin usar) para referencia/on-device edition; sus unit tests van skippidos.

## 4. Observaciones de depuración (por qué falló y cómo se arregló) — LEER ANTES DE TOCAR

1. **Windows cp1252 mata turnos LLM.** El server pipea stdout a un archivo; con `stdout.reconfigure(line_buffering=True)` sin `encoding`, CUALQUIER carácter fuera de latin-1 en un log (el `→` del print de timings, o un transcript español) lanzaba `UnicodeEncodeError` DENTRO de `run_turn` — después de generar la respuesta y ANTES del decisor — y el turno colapsaba a `released` silencioso. Toda la suite e2e falló por esto. Arreglo: `reconfigure(encoding="utf-8", errors="replace")` en stdout y stderr + `PYTHONUTF8=1` en el hijo de la suite. Lección: en Windows, toda impresión de texto del usuario es hostil al codepage.

2. **Semántica de frames del protocolo cloud.** El transcript SIEMPRE llega por la frame `transcript` temprana; `turn_final.transcription` casi siempre es `null`. Cualquier consumidor (tests util y browser `app.js`) que asigne el valor de `turn_final` sin chequear `is not null` pisa la transcripción correcta. Ambos lados ya están arreglados; si añades otro consumidor, replica esa regla.

3. **Echo-guard vs. transcripción en el mismo mensaje.** En el protocolo cloud, el transcript del usuario vive en las MISMAS parts del mensaje user que la instrucción del turno. El guard `echoes_instruction` sniffeaba `_instruction_text(messages)` y comparaba la respuesta contra el transcript — una traducción que re-expresa inglés ("What is the capital of France?") ES verbatim las palabras del usuario y era suprimida → fallback en modo translate. Arreglo: `run_turn` recibe `instruction=` explícito (la instrucción del turno, sin el transcript). NO retires el parámetro.

4. **Smart-turn + síntesis del fixture.** El clasificador es prosódico: el mismo texto con otra voz de OpenAI TTS ⇒ otro veredicto. El fixture `long_question` (voz por defecto → alloy) leía HOLD (p≈0.11) cuando la suite esperaba complete, y `long_question_male` (→ echo) leía complete cuando esperaba HOLD. Método reproducible: `benchmarks/voice_probe.py` (ordena voces por p(complete) sobre un texto) + `benchmarks/check_turns.py` (clasifica todo el set de fixtures) — no adivines: mide.

5. **Tests que describen infraestructura muerta.** Cualquier test contando "Primed cache" o asumiendo child llama-server pertenece a la era on-device: skip con motivo o rediseña el propósito real (chunked-merge ⇒ transcript correcto).

6. **Strict JSON schema de OpenAI.** `response_format=json_schema` con `strict: true` exige `additionalProperties: false` y required completos; `cloud._chat_body` lo añade. Si cambias `actions.HEAD_SCHEMA`, mantén esas invariantes (y recuerda que el fallback sin schema usa `cloud.parse_json` tolerante).

7. **Caché HF en Windows.** Nunca dejes que huggingface-hub escriba fuera de `parlor/.hf-cache/` (los env ya se fijan en `turn_detector.py`); si añades descargas de modelos, defaultea todos los envs HF ahí.

8. **La suite consume clave real.** Solo el RAZONADOR está mockeado (`tests/conftest.py`, trigger "stock"→500, "naples"→stall 8 s). Chat/STT/TTS/STT speech usan la clave de `.env`: cada vuelta completa de la suite cuesta fracciones de dólar y tarda ~9 min. Sin `OPENAI_API_KEY` los tests e2e se saltan (unit tests corren siempre). `PARLOR_TEST_URL` apunta la suite a un server externo.

9. **Pendiente de limpieza:** el commit `9578675` incluyó por accidente logs de pytest (`pytest-of-JuLiOBM/**`) — borra ese directorio del repo y añade `pytest-of-*/` a `.gitignore` (TMP de este host aterrizó dentro del workspace).

## 5. Estado al cierre

- Suite: la corrida e2e completa dio 75 passed / 3 failed; los 3 se arreglaron después (2 de translation por §4.3/§4.4 + 1 skip por llama de §4.5). Los 3 revisados PASARON en verificación focalizada (`test_translation.py`, `test_stream_parser.py`, `test_z_failure.py`: 22 passed, 1 skipped). La re-corrida completa se canceló por reinicio del server — relanzar `uv run pytest -q` para confirmar.
- Round-trips validados con la clave del mantenedor: chat streaming, TTS (OpenAI), y STT español ("Hola, esto es una prueba.").
- Costes con defaults (gpt-4o-mini + APIs OpenAI speech): del orden de **1–2 US cents/min de conversación**, dominado por STT/TTS; modelos de chat mayores escalan ese costo.
