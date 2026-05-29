@echo off
echo.
echo  FINANOM — Dashboard de Auditoria
echo  =================================
echo  Iniciando servidor local...
echo.
cd /d "%~dp0"
echo  Abre tu navegador en:  http://localhost:8080
echo.
echo  (Presiona Ctrl+C para detener)
echo.
uv run python -m http.server 8080
pause
