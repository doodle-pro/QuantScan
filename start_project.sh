#!/bin/bash

echo ""
echo "========================================"
echo "🧬 Q-MediScan: Quantum Cancer Detection"
echo "========================================"
echo ""
echo "Starting Q-MediScan development servers..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python not found. Please install Python 3.8+ and try again."
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 16+ and try again."
    exit 1
fi

echo "✅ Python and Node.js detected"
echo ""

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down Q-MediScan servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start backend server
echo "🔬 Starting Backend Server (FastAPI + Quantum ML)..."
cd "$DIR/backend"
$PYTHON_CMD run.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend server
echo "🎨 Starting Frontend Server (React + TypeScript)..."
cd "$DIR/frontend"
npm start &
FRONTEND_PID=$!

echo ""
echo "========================================"
echo "🚀 Q-MediScan is running!"
echo "========================================"
echo ""
echo "📊 Backend API: http://localhost:8000"
echo "🎨 Frontend UI: http://localhost:3000"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""
echo "⏳ Please wait for both servers to fully start..."
echo "🌐 Your browser should open automatically"
echo ""
echo "Press Ctrl+C to stop all servers..."

# Wait for background processes
wait $BACKEND_PID $FRONTEND_PID