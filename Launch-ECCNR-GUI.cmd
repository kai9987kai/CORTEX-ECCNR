@echo off
cd /d "%~dp0"
echo Starting CORTEX-ECCNR Interactive Scientific Observatory...
python scripts\launch_gui.py --desktop
pause
