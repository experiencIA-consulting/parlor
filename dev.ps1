# Parlor dev runner - start/stop/restart the server reliably, from any cwd.
#
#   .\dev.ps1            # restart (default)
#   .\dev.ps1 start      # start detached (logs -> server.log, survives the terminal)
#   .\dev.ps1 stop       # stop whatever listens on $Port
#   .\dev.ps1 status     # port + HTTP probe
#   .\dev.ps1 logs       # tail the log live (Ctrl+C to leave)
#   .\dev.ps1 restart -Host 0.0.0.0   # expose on the LAN (mobile testing)
#
# PARLOR_HOST defaults to 0.0.0.0 here (dev script, mobile-first); a value
# already exported in the environment or .env wins. PORT defaults to 8000.
# Windows PowerShell 5.1 and PowerShell 7 both work.

param(
    [ValidateSet("start", "stop", "restart", "status", "logs")]
    [string]$Action = "restart",
    [string]$Port = "",
    [Alias("Host")]
    [string]$HostAddress = ""
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$LogOut = Join-Path $Root "server.log"
$LogErr = Join-Path $Root "server.err.log"

# --- .env reader (no exporting, just PORT/PARLOR_HOST) ---
function Read-DotEnv([string]$key) {
    $envFile = Join-Path $Root ".env"
    if (-not (Test-Path $envFile)) { return $null }
    foreach ($line in Get-Content $envFile) {
        if ($line -match "^\s*$([regex]::Escape($key))\s*=\s*(.+?)\s*$") {
            return $Matches[1].Trim('"', "'")
        }
    }
    return $null
}

if (-not $Port) {
    if ($env:PORT) { $Port = $env:PORT }
    else {
        $v = Read-DotEnv "PORT"
        $Port = if ($v) { $v } else { "8000" }
    }
}
if (-not $HostAddress) {
    if ($env:PARLOR_HOST) { $HostAddress = $env:PARLOR_HOST }
    else {
        $v = Read-DotEnv "PARLOR_HOST"
        $HostAddress = if ($v) { $v } else { "0.0.0.0" }
    }
}
$PortNum = [int]$Port

# --- port helpers ---
function Get-ListenerPids {
    $pids = @()
    try {
        $pids = Get-NetTCPConnection -LocalPort $PortNum -State Listen -ErrorAction Stop |
            Select-Object -ExpandProperty OwningProcess -Unique
    } catch {
        # Fallback: netstat (some constrained shells lack Get-NetTCPConnection)
        foreach ($line in (netstat -ano | Select-String ":$PortNum\s")) {
            if ($line -match "\sLISTENING\s+(\d+)\s*$") { $pids += [int]$Matches[1] }
        }
    }
    return @($pids | Where-Object { $_ -gt 0 } | Select-Object -Unique)
}

function Test-Port {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", $PortNum)
        $ok = $c.Connected
        $c.Close()
        return $ok
    } catch { return $false }
}

function Stop-Parlor {
    $pids = Get-ListenerPids
    if (-not $pids) { Write-Host "nothing listening on $PortNum"; return }
    foreach ($p in $pids) {
        try { Stop-Process -Id $p -Force -ErrorAction Stop; Write-Host "stopped PID $p" }
        catch { Write-Host "could not stop PID ${p}: $($_.Exception.Message)" }
    }
    # wait for the port to free (TIME_WAIT / slow shutdown)
    for ($i = 0; $i -lt 20; $i++) {
        if (-not (Test-Port)) { break }
        Start-Sleep -Milliseconds 250
    }
}

function Start-Parlor {
    if (Test-Port) { Write-Host "already listening on $PortNum - run '.\dev.ps1 restart'"; return }
    $env:PORT = "$PortNum"
    $env:PARLOR_HOST = $HostAddress
    if (-not (Test-Path (Join-Path $Root "uv.lock"))) { throw "no uv.lock here - is $Root the parlor repo?" }
    Write-Host "starting: PORT=$PortNum PARLOR_HOST=$HostAddress"
    # Detached + logged: survives this terminal and this agent's session.
    Start-Process -FilePath "uv" -ArgumentList "run", "parlor" `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $LogOut -RedirectStandardError $LogErr `
        -WindowStyle Hidden
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 500
        if (Test-Port) { Write-Host "up: http://localhost:$PortNum  (logs: $LogOut)"; return }
    }
    throw "server did not come up in 20s - check $LogErr / $LogOut"
}

function Show-Status {
    $on = Test-Port
    Write-Host ("port {0}: {1}" -f $PortNum, $(if ($on) { "LISTENING" } else { "closed" }))
    if ($on) {
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$PortNum/" -UseBasicParsing -TimeoutSec 5
            Write-Host "HTTP /: $($r.StatusCode) $($r.StatusDescription)"
        } catch { Write-Host "HTTP /: $($_.Exception.Message)" }
    }
}

switch ($Action) {
    "stop"    { Stop-Parlor }
    "start"   { Start-Parlor }
    "restart" { Stop-Parlor; Start-Parlor }
    "status"  { Show-Status }
    "logs"    {
        if (Test-Path $LogOut) { Get-Content $LogOut -Tail 30 -Wait }
        else { Write-Host "no log yet at $LogOut" }
    }
}

