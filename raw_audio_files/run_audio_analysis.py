#!/usr/bin/env python3
"""
Audio Analysis Runner
Main script to run all three audio processing approaches:
1. Pyannote Speaker Recognition (most interesting to you)
2. SpeechBrain Speaker Recognition
3. Whisper Speech Recognition

Usage:
    python run_audio_analysis.py --all                    # Run all three
    python run_audio_analysis.py --pyannote               # Run only pyannote
    python run_audio_analysis.py --speechbrain            # Run only speechbrain  
    python run_audio_analysis.py --whisper                # Run only whisper
    python run_audio_analysis.py --help                   # Show help
"""

import argparse
import sys
import os
import time
from datetime import datetime

def check_dependencies():
    """Check if required dependencies are installed"""
    missing_deps = []
    
    # Check common dependencies
    try:
        import torch
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:
        missing_deps.append(f"Common ML libraries: {e}")
    
    # Check pyannote dependencies
    try:
        from pyannote.audio import Audio
        pyannote_available = True
    except ImportError:
        pyannote_available = False
        missing_deps.append("pyannote.audio")
    
    # Check speechbrain dependencies
    try:
        import speechbrain as sb
        speechbrain_available = True
    except ImportError:
        speechbrain_available = False
        missing_deps.append("speechbrain")
    
    # Check whisper dependencies
    try:
        import whisper
        whisper_available = True
    except ImportError:
        whisper_available = False
        missing_deps.append("openai-whisper")
    
    if missing_deps:
        print("Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall commands:")
        if not pyannote_available:
            print("  pip install pyannote.audio torchaudio")
        if not speechbrain_available:
            print("  pip install speechbrain")
        if not whisper_available:
            print("  pip install openai-whisper")
        print("  pip install torch numpy pandas matplotlib seaborn scikit-learn tqdm")
        return False
    
    return True

def run_pyannote(audio_dir=None):
    """Run Pyannote speaker recognition"""
    print("\n" + "="*60)
    print("RUNNING PYANNOTE SPEAKER RECOGNITION")
    print("="*60)
    
    try:
        from pyannote_speaker_recognition import main as pyannote_main
        start_time = time.time()
        pyannote_main(audio_dir)
        end_time = time.time()
        print(f"\nPyannote analysis completed in {end_time - start_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"Error running Pyannote: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_speechbrain(audio_dir=None):
    """Run SpeechBrain speaker recognition"""
    print("\n" + "="*60)
    print("RUNNING SPEECHBRAIN SPEAKER RECOGNITION")
    print("="*60)
    
    try:
        from speechbrain_speaker_recognition import main as speechbrain_main
        start_time = time.time()
        speechbrain_main(audio_dir)
        end_time = time.time()
        print(f"\nSpeechBrain analysis completed in {end_time - start_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"Error running SpeechBrain: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_whisper(audio_dir=None):
    """Run Whisper speech recognition"""
    print("\n" + "="*60)
    print("RUNNING WHISPER SPEECH RECOGNITION")
    print("="*60)
    
    try:
        from whisper_speech_recognition import main as whisper_main
        start_time = time.time()
        whisper_main(audio_dir)
        end_time = time.time()
        print(f"\nWhisper analysis completed in {end_time - start_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"Error running Whisper: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run audio analysis with different ML approaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_audio_analysis.py --all          # Run all three approaches
    python run_audio_analysis.py --pyannote     # Run only pyannote (recommended first)
    python run_audio_analysis.py --speechbrain  # Run only speechbrain
    python run_audio_analysis.py --whisper      # Run only whisper
        """
    )
    
    parser.add_argument("--all", action="store_true", 
                       help="Run all three approaches")
    parser.add_argument("--pyannote", action="store_true", 
                       help="Run Pyannote speaker recognition")
    parser.add_argument("--speechbrain", action="store_true", 
                       help="Run SpeechBrain speaker recognition")
    parser.add_argument("--whisper", action="store_true", 
                       help="Run Whisper speech recognition")
    parser.add_argument("--check-deps", action="store_true", 
                       help="Check if dependencies are installed")
    parser.add_argument("--audio-dir", type=str, default="raw_voice_recordings",
                       help="Directory containing audio files to analyze")
    
    args = parser.parse_args()
    
    # Show help if no arguments
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    print("="*60)
    print("AUDIO ANALYSIS RUNNER")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Check dependencies
    if args.check_deps:
        print("Checking dependencies...")
        if check_dependencies():
            print("✅ All dependencies are installed!")
        else:
            print("❌ Some dependencies are missing. Please install them first.")
        return
    
    if not check_dependencies():
        print("❌ Dependencies missing. Use --check-deps to see what's needed.")
        return
    
    # Check if audio directory exists
    audio_dir = args.audio_dir
    if not os.path.exists(audio_dir):
        print(f"❌ Audio directory '{audio_dir}' not found!")
        print("Make sure you're running this script from the correct directory.")
        return
    
    # Count audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.m4a', '.wav'))]
    print(f"📁 Found {len(audio_files)} audio files in {audio_dir}")
    
    if len(audio_files) == 0:
        print("❌ No audio files found!")
        return
    
    # Run selected approaches
    total_start_time = time.time()
    results = {}
    
    if args.all or args.pyannote:
        print("\n🎯 Starting with Pyannote (your preferred choice)...")
        results['pyannote'] = run_pyannote(audio_dir)
    
    if args.all or args.speechbrain:
        print("\n🧠 Running SpeechBrain...")
        results['speechbrain'] = run_speechbrain(audio_dir)
    
    if args.all or args.whisper:
        print("\n🎤 Running Whisper...")
        results['whisper'] = run_whisper(audio_dir)
    
    # Summary
    total_end_time = time.time()
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    print(f"Total runtime: {total_end_time - total_start_time:.2f} seconds")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nResults summary:")
    for approach, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {approach.title()}: {status}")
    
    print("\nGenerated files:")
    files_to_check = [
        "pyannote_confusion_matrix.png",
        "speechbrain_confusion_matrix.png", 
        "whisper_confusion_matrix.png",
        "whisper_transcriptions.json"
    ]
    
    for filename in files_to_check:
        if os.path.exists(filename):
            print(f"  ✅ {filename}")
    
    print(f"\n🎉 Analysis complete! Check the confusion matrices and results above.")
    
    if args.pyannote or args.all:
        print("\n💡 Since you were most interested in Pyannote, check the pyannote results first!")

if __name__ == "__main__":
    main()
