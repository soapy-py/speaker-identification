#!/usr/bin/env python3
"""
Audio Format Converter
Converts M4A files to WAV format for better compatibility with audio analysis libraries.
"""

import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm

def check_ffmpeg():
    """Check if ffmpeg is installed."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def convert_m4a_to_wav(input_path, output_path):
    """Convert M4A file to WAV using ffmpeg."""
    try:
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', '16000',          # 16kHz sample rate (good for speech)
            '-ac', '1',              # Mono
            '-y',                    # Overwrite output file
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr
            
    except Exception as e:
        return False, str(e)

def main():
    print("🔄 Audio Format Converter")
    print("=" * 50)
    
    # Check if ffmpeg is available
    if not check_ffmpeg():
        print("❌ ffmpeg is not installed or not found in PATH.")
        print("Please install ffmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/")
        sys.exit(1)
    
    print("✅ ffmpeg found")
    
    # Set up directories
    input_dir = Path("raw_voice_recordings")
    output_dir = Path("converted_wav_files")
    
    if not input_dir.exists():
        print(f"❌ Input directory '{input_dir}' not found")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Find all M4A files
    m4a_files = list(input_dir.glob("*.m4a"))
    
    if not m4a_files:
        print(f"❌ No M4A files found in '{input_dir}'")
        sys.exit(1)
    
    print(f"📁 Found {len(m4a_files)} M4A files")
    print(f"📁 Output directory: {output_dir}")
    
    # Convert files
    success_count = 0
    error_count = 0
    
    print("\n🔄 Converting files...")
    
    for m4a_file in tqdm(m4a_files, desc="Converting"):
        # Create output filename (replace .m4a with .wav)
        wav_filename = m4a_file.stem + ".wav"
        wav_path = output_dir / wav_filename
        
        # Convert
        success, error = convert_m4a_to_wav(m4a_file, wav_path)
        
        if success:
            success_count += 1
        else:
            error_count += 1
            print(f"\n❌ Failed to convert {m4a_file.name}: {error}")
    
    print(f"\n📊 Conversion Summary:")
    print(f"  ✅ Successfully converted: {success_count}")
    print(f"  ❌ Failed: {error_count}")
    print(f"  📁 WAV files saved to: {output_dir}")
    
    if success_count > 0:
        print(f"\n🎯 Next steps:")
        print(f"  1. Update your scripts to use '{output_dir}' instead of '{input_dir}'")
        print(f"  2. Or run: python run_audio_analysis.py --audio-dir {output_dir}")

if __name__ == "__main__":
    main()
