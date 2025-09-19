#!/usr/bin/env python3
"""
Pyannote Speaker Recognition Script
Adapted from the pyannote.ipynb notebook to work with M4A audio files
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    from pyannote.audio import Audio
    from pyannote.core import Segment
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
except ImportError:
    print("Error: pyannote.audio not installed. Please install with:")
    print("pip install pyannote.audio torchaudio scikit-learn matplotlib seaborn")
    exit(1)

class PyannoteConfig:
    """Configuration for pyannote speaker recognition"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "speechbrain/spkrec-ecapa-voxceleb"
        self.sample_rate = 16000
        self.mono = True
        self.test_size = 0.2
        self.random_state = 42
        self.max_iter = 1000

class PyannoteProcessor:
    def __init__(self, config):
        self.config = config
        print(f"Using device: {self.config.device}")
        
        # Initialize models
        self.embedding_model = PretrainedSpeakerEmbedding(
            self.config.model_name,
            device=self.config.device
        )
        self.audio_model = Audio(
            sample_rate=self.config.sample_rate, 
            mono=self.config.mono
        )
        
    def get_embedding(self, path):
        """Extract speaker embedding from audio file"""
        try:
            waveform, sample_rate = self.audio_model(path)
            tensor_waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(self.config.device)
            with torch.no_grad():
                return self.embedding_model(tensor_waveform).squeeze()
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None
    
    def build_dataset(self, base_dir):
        """Build dataset from audio files in directory"""
        data = []
        audio_files = [f for f in os.listdir(base_dir) if f.endswith(('.m4a', '.wav'))]
        
        print(f"Found {len(audio_files)} audio files")
        
        # Create user mapping from filenames
        users = set()
        for filename in audio_files:
            try:
                # Parse filename: {user_id}-{fatigue_index}-{timestamp}-{unique_hash}.m4a or .wav
                parts = filename.replace('.m4a', '').replace('.wav', '').split('-')
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
                parts = filename.replace('.m4a', '').replace('.wav', '').split('-')
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
        """Extract embeddings from all audio files in dataset"""
        print(f"Processing {len(df)} audio files...")
        
        embeddings, labels, user_ids = [], [], []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
            emb = self.get_embedding(row["path"])
            if emb is not None:
                embeddings.append(emb.cpu().numpy() if torch.is_tensor(emb) else emb)
                labels.append(row["label"])
                user_ids.append(row["user"])
            else:
                print(f"Skipping file due to error: {row['filename']}")
        
        if not embeddings:
            raise ValueError("No embeddings extracted successfully!")
        
        X = np.vstack(embeddings)
        y = np.array(labels)
        
        print(f"Successfully extracted {len(embeddings)} embeddings")
        print(f"Embedding shape: {X.shape}")
        
        return X, y, user_ids
    
    def train_classifier(self, X, y):
        """Train logistic regression classifier"""
        print("Training classifier...")
        
        # Check class distribution
        unique_classes, counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique_classes, counts))}")
        
        # Filter to only use classes with multiple samples for better training
        multi_sample_mask = np.isin(y, unique_classes[counts >= 2])
        if np.sum(multi_sample_mask) < len(y) * 0.5:  # If less than 50% have multiple samples
            print("Warning: Many users have only one sample. Using all data for training without stratification.")
            # Use simple train/test split without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=max(0.1, min(0.3, len(np.unique(y)) / len(y))),  # Adaptive test size
                random_state=self.config.random_state
            )
        else:
            # Use only users with multiple samples for stratified split
            X_filtered = X[multi_sample_mask]
            y_filtered = y[multi_sample_mask]
            print(f"Using {len(X_filtered)} samples from {len(np.unique(y_filtered))} users with multiple recordings")
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_filtered, y_filtered, 
                    stratify=y_filtered, 
                    test_size=self.config.test_size, 
                    random_state=self.config.random_state
                )
            except ValueError:
                # Fallback to non-stratified split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_filtered, y_filtered, 
                    test_size=self.config.test_size, 
                    random_state=self.config.random_state
                )
        
        clf = LogisticRegression(max_iter=self.config.max_iter).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        return clf, X_train, X_test, y_train, y_test, y_pred
    
    def evaluate_results(self, y_test, y_pred, user_ids, users):
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
        plt.title("Confusion Matrix - Pyannote Speaker Recognition")
        plt.tight_layout()
        plt.savefig("pyannote_confusion_matrix.png", dpi=300, bbox_inches='tight')
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
        """Run complete speaker recognition analysis"""
        start_time = time.time()
        
        print("=== Pyannote Speaker Recognition Analysis ===")
        print(f"Processing audio files from: {audio_dir}")
        
        # Build dataset - Updated to handle both M4A and WAV files
        df, users = self.build_dataset(audio_dir)
        if df.empty:
            print("No valid audio files found!")
            return
        
        print(f"\nDataset info:")
        print(f"Total files: {len(df)}")
        print(f"Users: {len(users)}")
        print(f"Files per user: {df.groupby('user').size().describe()}")
        
        # Extract embeddings
        X, y, user_ids = self.process_dataset(df)
        
        # Train classifier
        clf, X_train, X_test, y_train, y_test, y_pred = self.train_classifier(X, y)
        
        # Evaluate results
        accuracy, user_accuracy = self.evaluate_results(y_test, y_pred, user_ids, users)
        
        end_time = time.time()
        print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
        
        return {
            'accuracy': accuracy,
            'user_accuracy': user_accuracy,
            'classifier': clf,
            'dataset': df,
            'users': users,
            'processing_time': end_time - start_time
        }

def main(audio_dir=None):
    """Main function"""
    # Configuration
    config = PyannoteConfig()
    
    # Audio directory
    if audio_dir is None:
        audio_dir = "raw_voice_recordings"
    
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory '{audio_dir}' not found!")
        print("Make sure you're running this script from the correct directory.")
        return
    
    # Initialize processor
    processor = PyannoteProcessor(config)
    
    # Run analysis
    try:
        results = processor.run_analysis(audio_dir)
        if results:  # Check if results is not None
            print(f"\n=== Analysis Complete ===")
            print(f"Overall accuracy: {results['accuracy']:.2%}")
            print(f"Processing time: {results['processing_time']:.2f} seconds")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
