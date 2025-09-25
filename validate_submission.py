#!/usr/bin/env python3
"""
Validate submission files for Kaggle competition.
"""

import pandas as pd
from pathlib import Path


def validate_submission(submission_path: str, sample_path: str) -> bool:
    """Validate submission file format."""
    try:
        # Check for line number formatting issues first
        with open(submission_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith(('1.', '2.', '3.')) or any(first_line.startswith(f'{i}.') for i in range(1, 10)):
                print("❌ CSV file appears to have line numbers! First line:", first_line)
                print("💡 Tip: Use fix_csv_format.py to remove line numbers")
                return False
        
        # Load files
        submission = pd.read_csv(submission_path)
        sample = pd.read_csv(sample_path)
        
        print(f"Validating: {submission_path}")
        print(f"Submission shape: {submission.shape}")
        print(f"Sample shape: {sample.shape}")
        
        # Check columns
        if list(submission.columns) != list(sample.columns):
            print("❌ Column names don't match!")
            print(f"Submission columns: {list(submission.columns)}")
            print(f"Sample columns: {list(sample.columns)}")
            return False
        
        # Check IDs
        if not submission['id'].equals(sample['id']):
            print("❌ IDs don't match!")
            return False
        
        # Check for missing values
        if submission['BeatsPerMinute'].isnull().any():
            print("❌ Missing predictions found!")
            return False
        
        # Check prediction range (reasonable BPM values)
        min_bpm = submission['BeatsPerMinute'].min()
        max_bpm = submission['BeatsPerMinute'].max()
        mean_bpm = submission['BeatsPerMinute'].mean()
        
        print(f"BPM range: {min_bpm:.2f} - {max_bpm:.2f}")
        print(f"Mean BPM: {mean_bpm:.2f}")
        
        if min_bpm < 20 or max_bpm > 300:
            print("⚠️  Warning: BPM values outside typical range (20-300)")
        
        if min_bpm < 0:
            print("❌ Negative BPM values found!")
            return False
        
        print("✅ Submission format is valid!")
        return True
        
    except Exception as e:
        print(f"❌ Error validating submission: {e}")
        return False


def main():
    """Main validation function."""
    data_dir = Path("data")
    sample_path = data_dir / "sample_submission.csv"
    
    # Validate all submission files
    submission_files = [
        "submission.csv",
        "production_submission.csv",
        "final_submission.csv"
    ]
    
    print("=== SUBMISSION VALIDATION ===\n")
    
    for filename in submission_files:
        submission_path = data_dir / filename
        if submission_path.exists():
            validate_submission(submission_path, sample_path)
            print()
        else:
            print(f"❌ File not found: {filename}\n")
    
    print("=== VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()