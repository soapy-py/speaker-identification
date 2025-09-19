#!/usr/bin/env python3
"""
Whisper Speech Recognition Script
Adapted from the whisper.ipynb notebook to work with M4A audio files
"""

import os
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    import whisper
    import torch
except ImportError:
    print("Error: whisper or torch not installed. Please install with:")
    print("pip install openai-whisper torch torchaudio scikit-learn matplotlib seaborn")
    exit(1)

class WhisperConfig:
    """Configuration for Whisper speech recognition"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_size = "base"  # tiny, base, small, medium, large
        self.language = "en"  # Set to None for auto-detection
        self.test_size = 0.2
        self.random_state = 42
        self.max_iter = 1000
        self.max_features = 5000  # For TF-IDF

class WhisperProcessor:
    def __init__(self, config):
        self.config = config
        print(f"Using device: {self.config.device}")
        
        # Load Whisper model
        print(f"Loading Whisper {self.config.model_size} model...")
        self.model = whisper.load_model(self.config.model_size, device=self.config.device)
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
    def transcribe_audio(self, path):
        """Transcribe audio file to text"""
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                path, 
                language=self.config.language,
                fp16=False  # Use fp32 for better compatibility
            )
            
            return {
                'text': result['text'].strip(),
                'language': result['language'],
                'segments': len(result['segments']),
                'duration': sum(seg['end'] - seg['start'] for seg in result['segments'])
            }
        except Exception as e:
            print(f"Error transcribing {path}: {e}")
            return None
    
    def build_dataset(self, base_dir):
        """Build dataset from audio files in directory"""
        data = []
        audio_files = [f for f in os.listdir(base_dir) if f.endswith('.m4a')]
        
        print(f"Found {len(audio_files)} M4A files")
        
        # Create user mapping from filenames
        users = set()
        for filename in audio_files:
            try:
                # Parse filename: {user_id}-{fatigue_index}-{timestamp}-{unique_hash}.m4a
                parts = filename.replace('.m4a', '').split('-')
                if len(parts) >= 4:
                    user_id = parts[0]
                    users.add(user_id)
            except Exception as e:
                print(f"Error parsing filename {filename}: {e}")
        
        users = sorted(list(users))
        user_to_id = {user: idx for idx, user in enumerate(users)}
        
        print(f"Found {len(users)} unique users: {users}")
        
        for filename in audio_files:
            try:
                parts = filename.replace('.m4a', '').split('-')
                if len(parts) >= 4:
                    user_id = parts[0]
                    fatigue_index = parts[1]
                    timestamp = parts[2]
                    unique_hash = parts[3]
                    
                    data.append({
                        "path": os.path.join(base_dir, filename),
                        "label": user_to_id[user_id],
                        "user": user_id,
                        "fatigue_index": fatigue_index,
                        "timestamp": timestamp,
                        "filename": filename
                    })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        return pd.DataFrame(data), users
    
    def process_dataset(self, df):
        """Transcribe all audio files in dataset"""
        print(f"Transcribing {len(df)} audio files...")
        
        transcriptions = []
        valid_indices = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing audio"):
            result = self.transcribe_audio(row["path"])
            if result is not None and result['text']:
                transcriptions.append(result['text'])
                valid_indices.append(idx)
                
                # Add transcription info to the dataframe
                df.loc[idx, 'transcription'] = result['text']
                df.loc[idx, 'language'] = result['language']
                df.loc[idx, 'segments'] = result['segments']
                df.loc[idx, 'duration'] = result['duration']
            else:
                print(f"Skipping file due to transcription error: {row['filename']}")
        
        if not transcriptions:
            raise ValueError("No transcriptions extracted successfully!")
        
        # Filter dataframe to only include successfully transcribed files
        df_valid = df.loc[valid_indices].copy()
        
        print(f"Successfully transcribed {len(transcriptions)} files")
        print(f"Average transcription length: {np.mean([len(t.split()) for t in transcriptions]):.1f} words")
        
        return transcriptions, df_valid
    
    def extract_text_features(self, transcriptions):
        """Extract TF-IDF features from transcriptions"""
        print("Extracting text features...")
        
        # Fit and transform TF-IDF
        X = self.vectorizer.fit_transform(transcriptions)
        
        print(f"TF-IDF feature matrix shape: {X.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        return X.toarray()  # Convert sparse matrix to dense
    
    def train_classifier(self, X, y):
        """Train logistic regression classifier"""
        print("Training classifier...")
        
        # Check if we have enough samples per class for stratified split
        unique, counts = np.unique(y, return_counts=True)
        min_samples = min(counts)
        
        if min_samples < 2:
            print(f"Warning: Some classes have only {min_samples} sample(s). Using random split instead of stratified.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size, 
                random_state=self.config.random_state
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                stratify=y, 
                test_size=self.config.test_size, 
                random_state=self.config.random_state
            )
        
        clf = LogisticRegression(
            max_iter=self.config.max_iter, 
            random_state=self.config.random_state
        ).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        return clf, X_train, X_test, y_train, y_test, y_pred
    
    def analyze_transcriptions(self, df_valid):
        """Analyze transcription content"""
        print("\n=== Transcription Analysis ===")
        
        # Language distribution
        lang_dist = df_valid['language'].value_counts()
        print(f"Language distribution:\n{lang_dist}")
        
        # Duration statistics
        print(f"\nDuration statistics:")
        print(f"Mean: {df_valid['duration'].mean():.2f}s")
        print(f"Median: {df_valid['duration'].median():.2f}s")
        print(f"Min: {df_valid['duration'].min():.2f}s")
        print(f"Max: {df_valid['duration'].max():.2f}s")
        
        # Word count statistics
        df_valid['word_count'] = df_valid['transcription'].apply(lambda x: len(x.split()))
        print(f"\nWord count statistics:")
        print(f"Mean: {df_valid['word_count'].mean():.1f} words")
        print(f"Median: {df_valid['word_count'].median():.1f} words")
        print(f"Min: {df_valid['word_count'].min()} words")
        print(f"Max: {df_valid['word_count'].max()} words")
        
        # Sample transcriptions
        print(f"\nSample transcriptions:")
        for i, (_, row) in enumerate(df_valid.head(3).iterrows()):
            print(f"{i+1}. User: {row['user']}, Duration: {row['duration']:.2f}s")
            print(f"   Text: {row['transcription'][:100]}...")
    
    def evaluate_results(self, y_test, y_pred, df_valid, users):
        """Evaluate and display results"""
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix - Whisper Speech Recognition")
        plt.tight_layout()
        plt.savefig("whisper_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Per-user accuracy
        user_map = {i: user for i, user in enumerate(users)}
        df_eval = pd.DataFrame({'true': y_test, 'pred': y_pred})
        df_eval['user'] = df_eval['true'].map(user_map)
        df_eval['correct'] = df_eval['true'] == df_eval['pred']
        
        print("\nPer-user accuracy:")
        user_accuracy = df_eval.groupby('user')['correct'].mean()
        print(user_accuracy)
        
        return accuracy, user_accuracy
    
    def run_analysis(self, audio_dir):
        """Run complete speech recognition analysis"""
        start_time = time.time()
        
        print("=== Whisper Speech Recognition Analysis ===")
        print(f"Processing audio files from: {audio_dir}")
        
        # Build dataset
        df, users = self.build_dataset(audio_dir)
        if df.empty:
            print("No valid audio files found!")
            return
        
        print(f"\nDataset info:")
        print(f"Total files: {len(df)}")
        print(f"Users: {len(users)}")
        print(f"Files per user: {df.groupby('user').size().describe()}")
        
        # Transcribe audio files
        transcriptions, df_valid = self.process_dataset(df)
        
        # Analyze transcriptions
        self.analyze_transcriptions(df_valid)
        
        # Extract text features
        X = self.extract_text_features(transcriptions)
        y = df_valid['label'].values
        
        # Train classifier
        clf, X_train, X_test, y_train, y_test, y_pred = self.train_classifier(X, y)
        
        # Evaluate results
        accuracy, user_accuracy = self.evaluate_results(y_test, y_pred, df_valid, users)
        
        # Save transcriptions
        transcription_file = "whisper_transcriptions.json"
        transcription_data = df_valid[['filename', 'user', 'transcription', 'language', 'duration']].to_dict('records')
        with open(transcription_file, 'w') as f:
            json.dump(transcription_data, f, indent=2)
        print(f"\nTranscriptions saved to {transcription_file}")
        
        end_time = time.time()
        print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
        
        return {
            'accuracy': accuracy,
            'user_accuracy': user_accuracy,
            'classifier': clf,
            'vectorizer': self.vectorizer,
            'dataset': df_valid,
            'users': users,
            'processing_time': end_time - start_time
        }

def main():
    """Main function"""
    # Configuration
    config = WhisperConfig()
    
    # Audio directory
    audio_dir = "raw_voice_recordings"
    
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory '{audio_dir}' not found!")
        print("Make sure you're running this script from the correct directory.")
        return
    
    # Initialize processor
    processor = WhisperProcessor(config)
    
    # Run analysis
    try:
        results = processor.run_analysis(audio_dir)
        print(f"\n=== Analysis Complete ===")
        print(f"Overall accuracy: {results['accuracy']:.2%}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
