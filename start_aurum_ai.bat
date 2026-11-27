@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m streamlit run frontend/pages/NYSE_V3_Backtesting.py
pause
