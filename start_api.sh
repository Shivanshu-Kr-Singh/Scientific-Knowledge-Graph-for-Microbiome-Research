#!/bin/bash
#
# start_api.sh
# ------------
# Script to start the Scientific Knowledge Graph API server.
#
# Usage:
#   ./start_api.sh              # Start in development mode with auto-reload
#   ./start_api.sh --prod       # Start in production mode with 4 workers
#   ./start_api.sh --port 9000  # Start on custom port

set -e

# Default values
MODE="dev"
PORT=8000
WORKERS=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prod)
            MODE="prod"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --prod           Run in production mode with multiple workers"
            echo "  --port PORT      Port to run on (default: 8000)"
            echo "  --workers N      Number of workers for production mode (default: 4)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found"
    echo "Please create a .env file with Neo4j connection details:"
    echo ""
    echo "NEO4J_URI=bolt://localhost:7687"
    echo "NEO4J_USER=neo4j"
    echo "NEO4J_PASSWORD=your_password"
    echo ""
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if required packages are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Error: FastAPI is not installed"
    echo "Install dependencies with: pip install -r requirements.txt"
    exit 1
fi

# Start the server
echo "Starting Scientific Knowledge Graph API..."
echo "Mode: $MODE"
echo "Port: $PORT"

if [ "$MODE" = "prod" ]; then
    echo "Workers: $WORKERS"
    echo ""
    echo "API will be available at: http://localhost:$PORT"
    echo "Documentation: http://localhost:$PORT/docs"
    echo ""
    uvicorn api.query_api:app --host 0.0.0.0 --port "$PORT" --workers "$WORKERS"
else
    echo "Auto-reload: enabled"
    echo ""
    echo "API will be available at: http://localhost:$PORT"
    echo "Documentation: http://localhost:$PORT/docs"
    echo ""
    uvicorn api.query_api:app --host 0.0.0.0 --port "$PORT" --reload
fi
