@echo off
rem Parlor dev runner wrapper (batch is exempt from PowerShell execution
rem policy, which treats this workspace's scripts as untrusted):
rem   dev.cmd            -> restart
rem   dev.cmd start|stop|restart|status|logs [-Host 0.0.0.0] [-Port 8000]
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0dev.ps1" %*
