@echo off
REM Khoi chay ung dung Nhan dien bien so xe voi Python phu hop
CHCP 65001 >nul

SET VENV_SCRIPTS=venv\Scripts\python.exe
SET VENV_BIN=venv\bin\python.exe

IF EXIST %VENV_SCRIPTS% (
    %VENV_SCRIPTS% -c "import cv2, numpy, PIL, pytesseract" >nul 2>&1
    IF NOT ERRORLEVEL 1 (
        echo Dang khoi chay qua moi truong venv: %VENV_SCRIPTS%
        %VENV_SCRIPTS% app.py
        EXIT /B %ERRORLEVEL%
    )
)

IF EXIST %VENV_BIN% (
    %VENV_BIN% -c "import cv2, numpy, PIL, pytesseract" >nul 2>&1
    IF NOT ERRORLEVEL 1 (
        echo Dang khoi chay qua moi truong venv: %VENV_BIN%
        %VENV_BIN% app.py
        EXIT /B %ERRORLEVEL%
    )
)

echo Dang khoi chay qua Python he thong...
python app.py