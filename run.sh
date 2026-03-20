#!/bin/bash

# Native Start Script (using uv for Python and npm for Node)
# This will detect your hardware (NVIDIA or Apple Silicon) automatically.

# --- Configuration ---
BACKEND_DIR="backend"
FRONTEND_DIR="frontend"

# --- Colors ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}>>> Checking Python Environment (uv)...${NC}"

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not installed. Search online on how to install it, or"
    echo "Please install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Sync backend requirements into the environment
echo "Syncing backend dependencies..."
uv pip install -r "$BACKEND_DIR/requirements.txt"

echo -e "${GREEN}>>> Backend Ready (uv).${NC}"

echo -e "${BLUE}>>> Setting up Frontend (npm)...${NC}"
cd "$FRONTEND_DIR"
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install --silent
fi
cd ..

echo -e "${GREEN}>>> Frontend Ready.${NC}"

# Function to kill all background processes on exit
cleanup() {
    echo -e "\n${BLUE}>>> Stopping servers...${NC}"
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit
}

# On CTRL+C, make sure to close both servers, not just the script.
trap cleanup SIGINT

echo -e "${BLUE}>>> Starting Servers...${NC}"

# Run Backend with uv
echo "Starting Backend on http://localhost:9876..."
cd "$BACKEND_DIR"
uv run python app.py &
BACKEND_PID=$!
cd ..

# Run Frontend
echo "Starting Frontend on http://localhost:5173..."
cd "$FRONTEND_DIR"
npm run dev -- --clearScreen false &
FRONTEND_PID=$!
cd ..

echo -e "${GREEN}>>> BOTH SERVERS RUNNING!${NC}"
echo -e "${GREEN}>>> Access the app at http://localhost:5173${NC}"
echo -e "${BLUE}>>> (Press Ctrl+C to stop both servers)${NC}"

# Keep script alive
wait
