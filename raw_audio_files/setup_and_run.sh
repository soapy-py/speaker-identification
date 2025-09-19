#!/bin/bash

# Audio Analysis Setup and Run Script
# This script will set up the environment and run the analysis

echo "=========================================="
echo "Audio Analysis Setup & Run Script"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

echo "🐍 Using Python: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed or not in PATH"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📋 Installing requirements..."
pip install -r requirements.txt

# Check if audio files exist
if [ ! -d "raw_voice_recordings" ]; then
    echo "❌ Audio directory 'raw_voice_recordings' not found!"
    echo "Make sure you have your M4A files in the 'raw_voice_recordings' directory."
    exit 1
fi

# Count audio files
file_count=$(find raw_voice_recordings -name "*.m4a" | wc -l)
echo "📁 Found $file_count M4A files"

if [ $file_count -eq 0 ]; then
    echo "❌ No M4A files found in raw_voice_recordings directory!"
    exit 1
fi

# Check dependencies
echo "🔍 Checking dependencies..."
python3 run_audio_analysis.py --check-deps

if [ $? -ne 0 ]; then
    echo "❌ Dependency check failed. Please resolve the issues above."
    exit 1
fi

echo "✅ Setup complete!"
echo ""
echo "Choose what to run:"
echo "1. Run only Pyannote (recommended first - your preferred choice)"
echo "2. Run only SpeechBrain"
echo "3. Run only Whisper" 
echo "4. Run all three approaches"
echo "5. Exit"

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "🎯 Running Pyannote Speaker Recognition..."
        python3 run_audio_analysis.py --pyannote
        ;;
    2)
        echo "🧠 Running SpeechBrain Speaker Recognition..."
        python3 run_audio_analysis.py --speechbrain
        ;;
    3)
        echo "🎤 Running Whisper Speech Recognition..."
        python3 run_audio_analysis.py --whisper
        ;;
    4)
        echo "🚀 Running all three approaches..."
        python3 run_audio_analysis.py --all
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🎉 Analysis complete!"
echo ""
echo "Generated files:"
ls -la *.png *.json 2>/dev/null || echo "  No output files found."

echo ""
echo "To run again, use:"
echo "  source venv/bin/activate"
echo "  python3 run_audio_analysis.py --help"
