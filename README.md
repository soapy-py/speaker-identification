# Speaker Identification Project

A comprehensive audio analysis system for speaker recognition and speech transcription using multiple machine learning approaches.

## Overview

This project implements three different approaches to audio analysis:
1. **Pyannote** - Speaker recognition using pre-trained embeddings (Recommended)
2. **SpeechBrain** - Alternative speaker recognition approach
3. **Whisper** - OpenAI's speech-to-text transcription with speaker classification

The system can download audio files from AWS S3, process them through various ML models, and generate detailed analysis reports including confusion matrices and accuracy metrics.

## Key Features

- **Multi-modal Analysis**: Three different ML approaches for comprehensive audio analysis
- **AWS Integration**: Seamless audio file download from S3 with rate limiting
- **High Accuracy**: Achieved 96.77% accuracy with Pyannote approach
- **Batch Processing**: Handle individual files or bulk analysis
- **Comprehensive Reporting**: Generate confusion matrices, classification reports, and transcriptions

## Project Structure

```
├── raw_audio_files/           # Main Python analysis scripts
│   ├── pyannote_speaker_recognition.py
│   ├── speechbrain_speaker_recognition.py  
│   ├── whisper_speech_recognition.py
│   ├── run_audio_analysis.py  # Main runner script
│   ├── convert_audio.py
│   ├── setup_and_run.sh
│   ├── requirements.txt
│   └── index.ts               # TypeScript audio download system
├── jupyternotebook_scripts/   # Research notebooks
│   ├── pyannote.ipynb
│   ├── speechbrain_implementation.ipynb
│   └── whisper.ipynb
└── other supporting directories...
```

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js/Bun (for audio download system)
- AWS CLI configured with SSO
- Required Python packages (see requirements.txt)

### Installation

1. **Install Python dependencies:**
```bash
cd raw_audio_files
pip install -r requirements.txt
```

2. **Install Node.js dependencies:**
```bash
bun install  # or npm install
```

3. **Configure AWS credentials:**
```bash
aws configure sso
aws sso login
```

### Usage

#### Running Analysis

**Run all three approaches:**
```bash
python run_audio_analysis.py --all
```

**Run specific approach:**
```bash
python run_audio_analysis.py --pyannote     # Recommended
python run_audio_analysis.py --speechbrain  
python run_audio_analysis.py --whisper      
```

**Check dependencies:**
```bash
python run_audio_analysis.py --check-deps
```

#### Downloading Audio Files

**Sandbox mode:**
```bash
bun dev        # Uses sandbox.txt
bun dev --all  # Download all available recordings
```

**Production mode:**
```bash
bun start           # Uses production.txt
bun start --all     # Download all with rate limiting
```

## 🎵 Audio File Format

Audio files are organized with the naming convention:
```
{user_id}-{fatigue_index}-{timestamp}-{unique_hash}.m4a
```

## Model Performance

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Pyannote | 96.77% | Best performance, recommended |
| SpeechBrain | TBD | Alternative embedding approach |
| Whisper | TBD | Speech-to-text + classification |

## 🛠Technical Details

## 1. Pyannote (pyannote_speaker_recognition.py) ##
What it does:
•  Uses a pre-trained ECAPA-TDNN model (trained on VoxCeleb dataset)
•  Extracts speaker embeddings (numerical fingerprints) from audio
•  Trains a logistic regression classifier on these embeddings to identify speakers
•  No retraining of the deep model - just uses the pre-trained embeddings

About pretraining: The deep neural network is already trained on thousands of speakers from VoxCeleb. You're NOT retraining it - you're just using it as a feature extractor and training a simple classifier on top.

## 2. SpeechBrain (speechbrain_speaker_recognition.py) ## 
What it does:
•  Same concept as Pyannote but uses SpeechBrain's pre-trained models
•  Also extracts speaker embeddings from a pre-trained model
•  Trains logistic regression on the embeddings
•  No deep model retraining - just feature extraction + simple classification

## 3. Whisper (whisper_speech_recognition.py) ##
What it does:
•  Different approach entirely - uses speech content, not voice characteristics
•  Transcribes speech to text using Whisper
•  Uses TF-IDF (word frequency analysis) to classify speakers based on what they say
•  No voice characteristics - purely based on speech patterns and vocabulary

How they give probabilities:
All three output probabilities via the logistic regression classifier:
•  Input: Audio file
•  Output: Probability distribution across known speakers (e.g., 85% Speaker A, 15% Speaker B)

### AWS Integration
- PostgreSQL database integration for assessment ID lookup
- S3 file download with retry logic and rate limiting
- Configurable batch processing with delays

## Output Files

The system generates:
- **Confusion matrices** (PNG images)
- **Classification reports** (console output)
- **Transcription files** (JSON format)
- **Per-user accuracy metrics**

## Configuration

Key configuration options:
- Model selection (tiny, base, small, medium, large for Whisper)
- Batch processing limits and delays
- Test/train split ratios
- Feature extraction parameters

## Security Notes

- Audio files and credentials are excluded from git via .gitignore
- AWS SSO authentication required
- Sensitive configuration files are not committed

## Development

### Jupyter Notebooks
Prototyping and experimentation notebooks are available in `jupyternotebook_scripts/`:
- `pyannote.ipynb` - Pyannote model development
- `speechbrain_implementation.ipynb` - SpeechBrain experiments  
- `whisper.ipynb` - Whisper transcription testing

### Running Tests
```bash
python run_audio_analysis.py --check-deps
```
