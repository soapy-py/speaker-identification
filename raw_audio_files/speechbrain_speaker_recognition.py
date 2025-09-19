#!/usr/bin/env python3
"""
SpeechBrain Speaker Recognition Script
Adapted from the speechbrain_implementation.ipynb notebook to work with M4A audio files
"""

import os
import time
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    import speechbrain as sb
    from speechbrain.pretrained import EncoderClassifier
except ImportError:
    print("Error: speechbrain not installed. Please install with:")
    print("pip install speechbrain torchaudio scikit-learn matplotlib seaborn")
    exit(1)

class SpeechBrainConfig:
    """Configuration for SpeechBrain speaker recognition"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "speechbrain/spkrec-ecapa-voxceleb"
        self.sample_rate = 16000
        self.test_size = 0.2
        self.random_state = 42
        self.max_iter = 1000

class SpeechBrainProcessor:
    def __init__(self, config):
        self.config = config
        print(f"Using device: {self.config.device}")
        
        # Initialize SpeechBrain model
        print("Loading SpeechBrain model...")
        self.model = EncoderClassifier.from_hparams(
            source=self.config.model_name,
            run_opts={"device": self.config.device}
        )
        
    def load_audio(self, path):
        """Load and preprocess audio file"""
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(path)
            
            # Resample if necessary
            if sample_rate != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.config.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            return waveform.squeeze()
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def get_embedding(self, path):
        """Extract speaker embedding from audio file"""
        try:
            waveform = self.load_audio(path)
            if waveform is None:
                return None
            
            # Move to device
            waveform = waveform.to(self.config.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(waveform.unsqueeze(0))
                return embedding.squeeze().cpu()
        except Exception as e:
            print(f"Error processing {path}: {e}")
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
        """Extract embeddings from all audio files in dataset"""
        print(f"Processing {len(df)} audio files...")
        
        embeddings, labels, user_ids = [], [], []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
            emb = self.get_embedding(row["path"])
            if emb is not None:
                embeddings.append(emb.numpy() if torch.is_tensor(emb) else emb)
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
        plt.title("Confusion Matrix - SpeechBrain Speaker Recognition")
        plt.tight_layout()
        plt.savefig("speechbrain_confusion_matrix.png", dpi=300, bbox_inches='tight')
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
        
        print("=== SpeechBrain Speaker Recognition Analysis ===")
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

def main():
    """Main function"""
    # Configuration
    config = SpeechBrainConfig()
    
    # Audio directory
    audio_dir = "raw_voice_recordings"
    
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory '{audio_dir}' not found!")
        print("Make sure you're running this script from the correct directory.")
        return
    
    # Initialize processor
    processor = SpeechBrainProcessor(config)
    
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
