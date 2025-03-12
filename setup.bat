@echo off
setlocal

:: Step 1: Check if Python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python.
    echo Please visit https://www.python.org/downloads/ to download and install Python.
    exit /b 1
)

:: Step 2: Create virtual environment

:: Parse parameters
set force=0
if "%1"=="-f" (
    set force=1
)

if exist "venv" (
    if %force%==1 (
        echo Forcing removal of existing virtual environment...
        rmdir /s /q venv
        if %errorlevel% neq 0 (
            echo Failed to remove the virtual environment. Please check the error message.
            exit /b 1
        )
    ) else (
        echo Virtual environment 'venv' already exists.
    )
)

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. Please check the error message.
        exit /b 1
    )
    echo Virtual environment created successfully.
)

:: Step 3: Check if the activation script for the virtual environment exists
if not exist "venv\Scripts\activate" (
    echo Warning: The activation script 'venv\Scripts\activate' could not be found. Please check if the virtual environment was created correctly.
    exit /b 1
)

:: Step 4: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Step 5: Install Python packages from requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Failed to install dependencies. Please check the error message.
        exit /b 1
    )
) else (
    echo requirements.txt not found, skipping dependency installation.
)

echo Installation complete, all steps executed successfully.

endlocal
