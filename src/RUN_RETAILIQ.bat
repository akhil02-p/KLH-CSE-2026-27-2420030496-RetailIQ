@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul || (echo Python was not found in PATH.& pause& exit /b 1)
python -c "import streamlit" >nul 2>nul || (
  echo Streamlit is not installed in this Python environment.
  echo Run: python -m pip install -r requirements.txt
  pause
  exit /b 1
)
python -m streamlit run app.py
pause
