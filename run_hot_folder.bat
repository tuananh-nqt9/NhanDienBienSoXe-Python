@echo off
REM Khoi chay Thu muc nong (Hot Folder Watcher)
CHCP 65001 >nul

SET VENV_SCRIPTS=venv\Scripts\python.exe
SET VENV_BIN=venv\bin\python.exe

IF EXIST %VENV_SCRIPTS% (
    %VENV_SCRIPTS% -c "import cv2, numpy, PIL, pytesseract" >nul 2>&1
    IF NOT ERRORLEVEL 1 (
        echo Dang khoi chay qua moi truong venv: %VENV_SCRIPTS%
        %VENV_SCRIPTS% hot_folder.py
        EXIT /B %ERRORLEVEL%
    )
)

IF EXIST %VENV_BIN% (
    %VENV_BIN% -c "import cv2, numpy, PIL, pytesseract" >nul 2>&1
    IF NOT ERRORLEVEL 1 (
        echo Dang khoi chay qua moi truong venv: %VENV_BIN%
        %VENV_BIN% hot_folder.py
        EXIT /B %ERRORLEVEL%
    )
)

echo Dang khoi chay qua Python he thong...
python hot_folder.py
pause
