@echo off
REM Simple batch file to start YOLO RealSense Tracker on Windows startup
REM Assumes conda environment already exists

REM Configuration - Modify these paths as needed
set PROJECT_DIR=C:\Path\To\Your\Project
set CONDA_ENV_NAME=your_existing_env_name
set PYTHON_SCRIPT=project_with_evals.py

echo Starting YOLO RealSense Tracker startup script

REM Change to project directory
if exist "%PROJECT_DIR%" (
    cd /d "%PROJECT_DIR%"
    echo Changed to project directory: %PROJECT_DIR%
) else (
    echo ERROR: Project directory not found: %PROJECT_DIR%
    pause
    exit /b 1
)

REM Find conda installation
set CONDA_FOUND=0
for %%P in (
    "%USERPROFILE%\anaconda3\Scripts\activate.bat"
    "%USERPROFILE%\miniconda3\Scripts\activate.bat" 
    "%PROGRAMDATA%\Anaconda3\Scripts\activate.bat"
    "%PROGRAMDATA%\Miniconda3\Scripts\activate.bat"
) do (
    if exist "%%P" (
        set CONDA_ACTIVATE=%%P
        set CONDA_FOUND=1
        echo Found conda at: %%P
        goto :CondaFound
    )
)

:CondaFound
if %CONDA_FOUND%==0 (
    echo ERROR: Could not find conda installation
    pause
    exit /b 1
)

REM Activate the existing conda environment
echo Activating conda environment: %CONDA_ENV_NAME%
call "%CONDA_ACTIVATE%" "%CONDA_ENV_NAME%"
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment %CONDA_ENV_NAME%
    echo Make sure the environment exists and the name is correct
    pause
    exit /b 1
)

REM Verify Python script exists
if not exist "%PYTHON_SCRIPT%" (
    echo ERROR: Python script not found: %PYTHON_SCRIPT%
    pause
    exit /b 1
)

REM Wait for system stabilization after startup
echo Waiting 10 seconds for system stabilization...
timeout /t 10 /nobreak >nul

REM Run the tracker with the arguments from README example
echo Starting YOLO RealSense Tracker...

set ARGS=--rs-height 480 --rs-width 640 --rgb-exposure 1000 --stillness 1.0 --tilt 50 --conf 0.2 --max-distance 2.0 --min-distance 0.3 --decouple-segments 1 --decouple-zones "1:0.1-2.0,6.0-8.0" --no-video

echo Running: python %PYTHON_SCRIPT% %ARGS%

REM Start the tracker (remove 'start' to run in foreground, keep it to run in background)
start "YOLO Tracker" python "%PYTHON_SCRIPT%" %ARGS% >nul 2>nul

echo Tracker started
echo Startup script completed

REM Uncomment next line for debugging (keeps window open)
REM pause

exit /b 0