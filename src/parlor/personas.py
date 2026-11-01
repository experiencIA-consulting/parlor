"""Session personalities: voice + style presets the user can switch live.

A preset bundles THREE things the cloud stack styles per request:
- `voice`: OpenAI's TTS timbre (gpt-4o-mini-tts ships a fixed catalog:
  alloy/ash/ballad/coral/echo/fable/nova/onyx/sage/shimmer/verse — no
  voice cloning), "" keeps whatever OPENAI_TTS_VOICE/mode default is.
- `tts_instructions`: gpt-4o-mini-tts's `instructions` field — the
  DELIVERY directive (tone, pace, emotion, accent). This is where a
  "robotized" voice comes from; it rides the speech API, never the
  conversation model.
- `system`: one sentence appended to the session system prompt — the
  CHAT persona (what the text is like). Only ever read from here, so a
  preset change never touches turn prompts, and no freeform user text
  can inject itself into the prompt (curated server-side catalog by
  design).

Costs: the persona clause rides the system prompt (a few dozen tokens
per request); instructions ride the TTS input (character-priced, cents
per minute). voice_preview lets the user hear one short line before
switching — the only extra API call is that deliberate sample.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    key: str                  # stable id — what travels over the wire
    label: str                # shown in the UI (Spanish)
    description: str          # one-liner in the selector
    voice: str                # OpenAI voice id, "" = session default
    tts_instructions: str     # speech delivery (TTS instructions)
    system: str               # chat-persona clause ("" = the stock assistant)
    sample_line: str          # what voice_preview speaks


PRESETS = [
    Persona(
        key="neutral", label="Neutral", voice="",
        description="El asistente de siempre — voz y trato neutros",
        tts_instructions="Speak naturally, warm and even.",
        system="",  # the stock SYSTEM_PROMPT IS the neutral persona
        sample_line="Hola — así sueno normalmente. ¿En qué te ayudo?",
    ),
    Persona(
        key="narradora", label="La Narradora", voice="shimmer",
        description="Cuentacuentos cálida, con drama y pausas",
        tts_instructions=("Warm expressive storyteller voice, gentle pace, "
                          "light theatrical emphasis, clear enunciation."),
        system=("You speak like a warm storyteller: rich, a little "
                "theatrical, comfortable with vivid detail."),

        sample_line=("Calla, calla… dime si alguna vez te han contado la "
                     "historia de la zorra que aprendió a volar."),
    ),
    Persona(
        key="amigo", label="Amigo Casero", voice="sage",
        description="Mejor amigo relajado, informal y animado",
        tts_instructions="Relaxed friendly voice, upbeat, natural flow, not formal.",
        system=("You chat like a close friend: informal, "
                "direct, playful; short lively sentences, no lecture."),

        sample_line="Ey! ¿Qué onda? Cuéntame — ¿qué está pasando?",
    ),
    Persona(
        key="profesor", label="El Profesor", voice="onyx",
        description="Voz masculina serena, explica paso a paso",
        tts_instructions=("Calm authoritative male voice, measured pace, "
                          "crisp articulation."),
        system=("You explain like a patient professor: precise, ordered, "
                "with a concrete example before moving on."),
        sample_line="Bienvenido. ¿Repasamos el tema paso a paso, o prefieres lo esencial?",
    ),
    Persona(
        key="robot", label="Robot", voice="echo",
        description="Robótico: plano, sin emociones, te procesa",
        tts_instructions=("Flat robotic monotone delivery, no emotion, "
                          "even staccato rhythm, slightly clipped words."),
        system=("You answer like a helpful robot: matter-of-fact, dry, "
                "fascinated by efficiency — never emotes."),
        sample_line="Bip-bup. Función de asistente activa. Esperando instrucción.",
    ),
    Persona(
        key="noticias", label="Presentador/a", voice="nova",
        description="Conductor/a de noticias: objetivo y ágil",
        tts_instructions=("Crisp professional newsreader voice, brisk "
                          "neutral pace, polished diction."),
        system=("You answer like a news presenter: objective, concise "
                "sentences, facts first and no filler."),
        sample_line="Buenas noticias: aquí está lo que necesitas saber.",
    ),
    Persona(
        key="coach", label="El Coach", voice="coral",
        description="Energético y motivador, te empuja a actuar",
        tts_instructions=("Energetic motivational-coach voice, dynamic "
                          "pace, encouraging emphasis."),
        system=("You answer like a motivating coach: energizing, direct, "
                "always steering toward action."),
        sample_line="¡Órale! Hoy es el día — vamos con todo.",
    ),
]

_DEFAULT_KEY = "neutral"
_BY_KEY = {p.key: p for p in PRESETS}


def get(key: str) -> Persona:
    """Unknown keys degrade to neutral — a stale localStorage value or a
    half-typed UI must never strand the session without a persona."""
    return _BY_KEY.get(key.strip().lower(), _BY_KEY[_DEFAULT_KEY])
