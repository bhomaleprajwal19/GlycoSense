#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🩺 GlycoSense — Diabetes Risk Prediction"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ python3 not found. Please install Python 3.8+"
  exit 1
fi

# Install deps if needed
if ! python3 -c "import flask, sklearn, joblib" 2>/dev/null; then
  echo "📦 Installing dependencies..."
  pip3 install -r backend/requirements.txt
fi

# Train if models don't exist
if [ ! -f "backend/models/random_forest.pkl" ]; then
  echo "🤖 Training models (first-time setup)..."
  python3 train_models.py
  echo ""
fi

# Start API
echo "🚀 Starting API on http://localhost:8080..."
python3 backend/app.py &
API_PID=$!

sleep 1.5

# Open frontend
FRONTEND="$SCRIPT_DIR/frontend/index.html"
echo "🌐 Opening frontend: $FRONTEND"
if command -v open &>/dev/null; then
  open "$FRONTEND"
elif command -v xdg-open &>/dev/null; then
  xdg-open "$FRONTEND"
else
  echo "   Please open manually: $FRONTEND"
fi

echo ""
echo "✅ GlycoSense is running!"
echo "   API:      http://localhost:8080/api/health"
echo "   Frontend: $FRONTEND"
echo ""
echo "Press Ctrl+C to stop the server."
wait $API_PID
