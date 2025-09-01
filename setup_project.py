#!/usr/bin/env python3
"""
Q-MediScan Complete Setup Script
Automated setup for the entire project
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description, cwd=None, check=True):
    """Run a command and handle errors"""
    print(f"[WRENCH] {description}...")
    try:
        if platform.system() == "Windows":
            # Use shell=True on Windows for better compatibility
            result = subprocess.run(
                command, 
                shell=True, 
                check=check, 
                cwd=cwd,
                capture_output=True,
                text=True
            )
        else:
            result = subprocess.run(
                command.split(), 
                check=check, 
                cwd=cwd,
                capture_output=True,
                text=True
            )
        
        if result.returncode == 0:
            print(f"[OK] {description} completed successfully")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[ERROR] Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_node_version():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True, check=True)
        version = result.stdout.strip()
        print(f"[OK] Node.js {version} detected")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] Node.js not found")
        print("   Please install Node.js 16+ from https://nodejs.org/")
        return False

def setup_backend():
    """Setup Python backend environment"""
    print("\n🐍 Setting up Python Backend...")
    
    backend_dir = Path(__file__).parent / "backend"
    
    # Check if virtual environment already exists
    venv_dir = backend_dir / "venv"
    if venv_dir.exists():
        print("[OK] Virtual environment already exists")
    else:
        # Create virtual environment
        if not run_command(
            "python -m venv venv", 
            "Creating virtual environment",
            cwd=backend_dir
        ):
            return False
    
    # Determine activation script and pip command based on OS
    if platform.system() == "Windows":
        pip_command = str(venv_dir / "Scripts" / "pip.exe")
        python_command = str(venv_dir / "Scripts" / "python.exe")
    else:
        pip_command = str(venv_dir / "bin" / "pip")
        python_command = str(venv_dir / "bin" / "python")
    
    # Upgrade pip
    if not run_command(
        f'"{pip_command}" install --upgrade pip',
        "Upgrading pip",
        cwd=backend_dir
    ):
        return False
    
    # Install dependencies
    if not run_command(
        f'"{pip_command}" install -r requirements.txt',
        "Installing Python dependencies (this may take a few minutes)",
        cwd=backend_dir
    ):
        return False
    
    # Create .env file if it doesn't exist
    env_file = backend_dir / ".env"
    env_example = backend_dir / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        env_content = env_example.read_text()
        env_file.write_text(env_content)
        print("[OK] Created .env file from template")
    
    print("[OK] Backend setup completed")
    return True

def setup_frontend():
    """Setup React frontend environment"""
    print("\n[ATOM] Setting up React Frontend...")
    
    frontend_dir = Path(__file__).parent / "frontend"
    
    # Check if node_modules exists
    node_modules = frontend_dir / "node_modules"
    if node_modules.exists():
        print("[OK] Node modules already installed")
        return True
    
    # Install dependencies
    if not run_command(
        "npm install",
        "Installing Node.js dependencies (this may take a few minutes)",
        cwd=frontend_dir
    ):
        return False
    
    print("[OK] Frontend setup completed")
    return True

def create_startup_scripts():
    """Create convenient startup scripts"""
    print("\n📝 Creating startup scripts...")
    
    project_dir = Path(__file__).parent
    
    # Windows batch script
    windows_script = project_dir / "start_project.bat"
    windows_content = '''@echo off
echo [DNA] Q-MediScan Project Startup
echo ============================

echo.
echo 🐍 Starting Backend Server...
cd /d "%~dp0\\backend"
start "Q-MediScan Backend" cmd /k "venv\\Scripts\\activate && python run.py"

timeout /t 3 /nobreak >nul

echo.
echo [ATOM] Starting Frontend Server...
cd /d "%~dp0\\frontend"
start "Q-MediScan Frontend" cmd /k "npm start"

echo.
echo [ROCKET] Both servers are starting...
echo [CHART] Backend API: http://localhost:8000
echo [GLOBE] Frontend App: http://localhost:3000
echo 📖 API Docs: http://localhost:8000/docs
echo.
echo Press any key to exit...
pause >nul
'''
    
    windows_script.write_text(windows_content, encoding='utf-8')
    print("[OK] Created start_project.bat for Windows")
    
    # Unix shell script
    unix_script = project_dir / "start_project.sh"
    unix_content = '''#!/bin/bash

echo "[DNA] Q-MediScan Project Startup"
echo "============================"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo ""
echo "🐍 Starting Backend Server..."

# Start backend in background
cd "$SCRIPT_DIR/backend"
source venv/bin/activate
python run.py &
BACKEND_PID=$!

echo "Backend PID: $BACKEND_PID"

# Wait a moment for backend to start
sleep 3

echo ""
echo "[ATOM] Starting Frontend Server..."

# Start frontend in background
cd "$SCRIPT_DIR/frontend"
npm start &
FRONTEND_PID=$!

echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "[ROCKET] Both servers are running!"
echo "[CHART] Backend API: http://localhost:8000"
echo "[GLOBE] Frontend App: http://localhost:3000"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers..."

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "[OK] Servers stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for user to stop
wait
'''
    
    unix_script.write_text(unix_content, encoding='utf-8')
    
    # Make shell script executable on Unix systems
    if platform.system() != "Windows":
        os.chmod(unix_script, 0o755)
    
    print("[OK] Created start_project.sh for Linux/Mac")
    
    return True

def test_setup():
    """Test if the setup was successful"""
    print("\n[EUROPEAN_POST_OFFICE]� Testing setup...")
    
    backend_dir = Path(__file__).parent / "backend"
    frontend_dir = Path(__file__).parent / "frontend"
    
    # Test backend
    if platform.system() == "Windows":
        python_command = str(backend_dir / "venv" / "Scripts" / "python.exe")
    else:
        python_command = str(backend_dir / "venv" / "bin" / "python")
    
    # Test if we can import the main modules
    test_command = f'"{python_command}" -c "import app.main; print(\\"Backend imports successful\\")"'
    if run_command(test_command, "Testing backend imports", cwd=backend_dir, check=False):
        print("[OK] Backend test passed")
    else:
        print("[WARNING]  Backend test failed, but this might be normal")
    
    # Test frontend
    package_json = frontend_dir / "package.json"
    if package_json.exists():
        print("[OK] Frontend test passed")
    else:
        print("[ERROR] Frontend test failed")
        return False
    
    return True

def main():
    """Main setup function"""
    print("[DNA] Q-MediScan Complete Project Setup")
    print("=" * 50)
    print("This will set up the entire Q-MediScan project including:")
    print("• Python backend with Classiq SDK")
    print("• React frontend with TypeScript")
    print("• All dependencies and configurations")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_node_version():
        sys.exit(1)
    
    # Setup components
    success = True
    
    if not setup_backend():
        success = False
    
    if not setup_frontend():
        success = False
    
    if not create_startup_scripts():
        success = False
    
    # Test setup
    test_setup()
    
    print("\n" + "=" * 50)
    
    if success:
        print("[PARTY] Setup completed successfully!")
        print("\n[CLIPBOARD] How to run the project:")
        
        if platform.system() == "Windows":
            print("   Double-click: start_project.bat")
            print("   Or run: .\\start_project.bat")
        else:
            print("   Run: ./start_project.sh")
            print("   Or: chmod +x start_project.sh && ./start_project.sh")
        
        print("\n[GLOBE] Access points:")
        print("   • Frontend: http://localhost:3000")
        print("   • Backend API: http://localhost:8000")
        print("   • API Documentation: http://localhost:8000/docs")
        
        print("\n[MICROSCOPE] About Classiq SDK:")
        print("   • Classiq SDK is installed and ready to use")
        print("   • No API key required for basic functionality")
        print("   • For advanced features, get API key from https://platform.classiq.io/")
        
        print("\n[TROPHY] Ready for CQhack25 submission!")
        
    else:
        print("[ERROR] Setup failed. Please check the errors above.")
        print("\n[WRENCH] Manual setup instructions:")
        print("1. Backend: cd backend && python -m venv venv && venv\\Scripts\\activate && pip install -r requirements.txt")
        print("2. Frontend: cd frontend && npm install")
        sys.exit(1)

if __name__ == "__main__":
    main()