#!/usr/bin/env python3
"""
Script to download the Kaggle competition dataset.
For the playground-series-s5e9 competition.
"""

import os
import subprocess
import sys
from pathlib import Path


def setup_kaggle_credentials():
    """Set up Kaggle credentials if not already configured."""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # Check if credentials exist
    if not (kaggle_dir / 'kaggle.json').exists():
        print("Kaggle credentials not found.")
        print("Please set up your Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Create API Token")
        print("3. Place kaggle.json in ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Set permissions
    os.chmod(kaggle_dir / 'kaggle.json', 0o600)
    return True


def download_dataset():
    """Download the competition dataset."""
    try:
        # Create data directory
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        # Download competition data
        cmd = ['kaggle', 'competitions', 'download', '-c', 'playground-series-s5e9']
        result = subprocess.run(cmd, cwd=data_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error downloading dataset: {result.stderr}")
            return False
        
        # Extract if zip file exists
        import zipfile
        zip_path = data_dir / 'playground-series-s5e9.zip'
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            zip_path.unlink()  # Remove zip file
        
        print("Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def create_sample_data():
    """Create sample data for testing if real data is not available."""
    import pandas as pd
    import numpy as np
    
    print("Creating sample data for testing...")
    
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Create sample training data with audio features
    np.random.seed(42)
    n_train = 1000
    
    # Generate synthetic audio features that might correlate with BPM
    features = {
        'id': range(n_train),
        'tempo': np.random.normal(120, 30, n_train),  # tempo often correlates with BPM
        'loudness': np.random.normal(-10, 5, n_train),
        'energy': np.random.uniform(0, 1, n_train),
        'danceability': np.random.uniform(0, 1, n_train),
        'valence': np.random.uniform(0, 1, n_train),
        'acousticness': np.random.uniform(0, 1, n_train),
        'instrumentalness': np.random.uniform(0, 1, n_train),
        'liveness': np.random.uniform(0, 1, n_train),
        'speechiness': np.random.uniform(0, 1, n_train),
        'duration_ms': np.random.normal(200000, 50000, n_train),
        'key': np.random.randint(0, 12, n_train),
        'mode': np.random.randint(0, 2, n_train),
        'time_signature': np.random.choice([3, 4, 5], n_train, p=[0.1, 0.8, 0.1]),
    }
    
    # Create BPM target with some correlation to features
    bpm = (features['tempo'] * 0.8 + 
           features['energy'] * 20 + 
           features['danceability'] * 15 + 
           np.random.normal(0, 10, n_train))
    bpm = np.clip(bpm, 60, 200)  # Reasonable BPM range
    
    features['BeatsPerMinute'] = bpm
    
    train_df = pd.DataFrame(features)
    train_df.to_csv(data_dir / 'train.csv', index=False)
    
    # Create test data (without target)
    n_test = 300
    test_features = {
        'id': range(n_train, n_train + n_test),
        'tempo': np.random.normal(120, 30, n_test),
        'loudness': np.random.normal(-10, 5, n_test),
        'energy': np.random.uniform(0, 1, n_test),
        'danceability': np.random.uniform(0, 1, n_test),
        'valence': np.random.uniform(0, 1, n_test),
        'acousticness': np.random.uniform(0, 1, n_test),
        'instrumentalness': np.random.uniform(0, 1, n_test),
        'liveness': np.random.uniform(0, 1, n_test),
        'speechiness': np.random.uniform(0, 1, n_test),
        'duration_ms': np.random.normal(200000, 50000, n_test),
        'key': np.random.randint(0, 12, n_test),
        'mode': np.random.randint(0, 2, n_test),
        'time_signature': np.random.choice([3, 4, 5], n_test, p=[0.1, 0.8, 0.1]),
    }
    
    test_df = pd.DataFrame(test_features)
    test_df.to_csv(data_dir / 'test.csv', index=False)
    
    # Create sample submission
    sample_submission = pd.DataFrame({
        'id': test_df['id'],
        'BeatsPerMinute': np.full(n_test, 120.0)  # Default prediction
    })
    sample_submission.to_csv(data_dir / 'sample_submission.csv', index=False)
    
    print("Sample data created successfully!")
    return True


def main():
    """Main function to download dataset."""
    print("Setting up Kaggle competition data download...")
    
    # Try to set up credentials and download real data
    if setup_kaggle_credentials():
        if download_dataset():
            return
    
    # If real data download fails, create sample data
    print("Unable to download real data, creating sample data for development...")
    create_sample_data()


if __name__ == "__main__":
    main()