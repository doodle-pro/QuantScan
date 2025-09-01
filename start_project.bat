@echo off
echo.
echo ========================================
echo 🧬 Q-MediScan: Quantum Cancer Detection
echo ========================================
echo.
echo Starting Q-MediScan development servers...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found. Please install Node.js 16+ and try again.
    pause
    exit /b 1
)

echo ✅ Python and Node.js detected
echo.

REM Start backend server
echo 🔬 Starting Backend Server (FastAPI + Quantum ML)...
echo.
start "Q-MediScan Backend" cmd /k "cd /d %~dp0backend && python run.py"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend server
echo 🎨 Starting Frontend Server (React + TypeScript)...
echo.
start "Q-MediScan Frontend" cmd /k "cd /d %~dp0frontend && npm start"

echo.
echo ========================================
echo 🚀 Q-MediScan is starting up!
echo ========================================
echo.
echo 📊 Backend API: http://localhost:8000
echo 🎨 Frontend UI: http://localhost:3000
echo 📖 API Docs: http://localhost:8000/docs
echo.
echo ⏳ Please wait for both servers to fully start...
echo 🌐 Your browser should open automatically
echo.
echo Press any key to close this window...
pause >nul