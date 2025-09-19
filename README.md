# Speaker Identification Project

A comprehensive audio analysis system for speaker recognition and speech transcription using multiple machine learning approaches.

## Overview

This project implements three different approaches to audio analysis:
1. **Pyannote** - Speaker recognition using pre-trained embeddings (Recommended)
2. **SpeechBrain** - Alternative speaker recognition approach
3. **Whisper** - OpenAI's speech-to-text transcription with speaker classification

The system can download audio files from AWS S3, process them through various ML models, and generate detailed analysis reports including confusion matrices and accuracy metrics.

## 🎯 Key Features

- **Multi-modal Analysis**: Three different ML approaches for comprehensive audio analysis
- **AWS Integration**: Seamless audio file download from S3 with rate limiting
- **High Accuracy**: Achieved 96.77% accuracy with Pyannote approach
- **Batch Processing**: Handle individual files or bulk analysis
- **Comprehensive Reporting**: Generate confusion matrices, classification reports, and transcriptions

## 📁 Project Structure

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

## 🚀 Quick Start

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

## 📊 Model Performance

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Pyannote | 96.77% | Best performance, recommended |
| SpeechBrain | TBD | Alternative embedding approach |
| Whisper | TBD | Speech-to-text + classification |

## 🛠️ Technical Details

### Pyannote Approach
- Uses pre-trained ECAPA-TDNN embeddings
- Logistic regression classifier on speaker embeddings
- Handles variable sample sizes with adaptive splitting

### SpeechBrain Approach  
- Similar embedding-based approach with different pre-trained models
- Compatible with Pyannote pipeline

### Whisper Approach
- OpenAI Whisper for speech transcription
- TF-IDF feature extraction from transcriptions
- Text-based speaker classification

### AWS Integration
- PostgreSQL database integration for assessment ID lookup
- S3 file download with retry logic and rate limiting
- Configurable batch processing with delays

## 📈 Output Files

The system generates:
- **Confusion matrices** (PNG images)
- **Classification reports** (console output)
- **Transcription files** (JSON format)
- **Per-user accuracy metrics**

## ⚙️ Configuration

Key configuration options:
- Model selection (tiny, base, small, medium, large for Whisper)
- Batch processing limits and delays
- Test/train split ratios
- Feature extraction parameters

## 🔒 Security Notes

- Audio files and credentials are excluded from git via .gitignore
- AWS SSO authentication required
- Sensitive configuration files are not committed

## 📝 Development

### Jupyter Notebooks
Prototyping and experimentation notebooks are available in `jupyternotebook_scripts/`:
- `pyannote.ipynb` - Pyannote model development
- `speechbrain_implementation.ipynb` - SpeechBrain experiments  
- `whisper.ipynb` - Whisper transcription testing

### Running Tests
```bash
python run_audio_analysis.py --check-deps
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎉 Acknowledgments

- OpenAI Whisper team for the speech recognition model
- Pyannote.audio team for speaker embedding models
- SpeechBrain team for the toolkit and pre-trained models
