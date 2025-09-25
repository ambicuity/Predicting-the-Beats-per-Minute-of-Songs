#!/usr/bin/env python3
"""
Fix CSV format issues by removing line numbers and ensuring proper format.
"""

import pandas as pd
import re
from pathlib import Path

def fix_csv_line_numbers(file_path: str) -> bool:
    """
    Fix CSV files that have line numbers at the beginning of each line.
    
    Args:
        file_path: Path to the CSV file to fix
        
    Returns:
        bool: True if file was fixed, False if no fix was needed
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"Checking {file_path}...")
    
    # Read raw content
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        print("❌ File is empty")
        return False
    
    # Check if first line has line numbers
    first_line = lines[0].strip()
    
    # Pattern to detect line numbers at the start: "1.text" or "1text" 
    if re.match(r'^\d+\.', first_line):
        print(f"🔧 Detected line numbers in file. First line: '{first_line}'")
        
        # Create backup
        backup_path = str(file_path) + '.backup'
        with open(backup_path, 'w') as f:
            f.writelines(lines)
        print(f"📋 Backup created: {backup_path}")
        
        # Fix all lines by removing line numbers
        fixed_lines = []
        for line in lines:
            # Remove pattern like "1." or "123." from the beginning
            fixed_line = re.sub(r'^\d+\.', '', line)
            fixed_lines.append(fixed_line)
        
        # Write fixed content
        with open(file_path, 'w') as f:
            f.writelines(fixed_lines)
        
        print(f"✅ Fixed file. Removed line numbers from {len(fixed_lines)} lines.")
        
        # Verify the fix
        try:
            df = pd.read_csv(file_path)
            if 'id' in df.columns:
                print(f"✅ Verification successful: 'id' column found. Shape: {df.shape}")
            else:
                print(f"⚠️  Warning: 'id' column still not found. Columns: {list(df.columns)}")
        except Exception as e:
            print(f"❌ Error verifying fixed file: {e}")
            
        return True
    else:
        # Check if file can be read properly
        try:
            df = pd.read_csv(file_path)
            if 'id' in df.columns:
                print(f"✅ File is already in correct format. Shape: {df.shape}")
            else:
                print(f"⚠️  File has no line numbers but 'id' column not found. Columns: {list(df.columns)}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            
        return False

def main():
    """Fix CSV format issues in submission files."""
    print("=== CSV FORMAT FIXER ===\n")
    
    data_dir = Path("data")
    submission_files = [
        "final_submission.csv",
        "submission.csv", 
        "production_submission.csv"
    ]
    
    fixed_count = 0
    for filename in submission_files:
        file_path = data_dir / filename
        if file_path.exists():
            if fix_csv_line_numbers(file_path):
                fixed_count += 1
            print()
        else:
            print(f"⏩ Skipping {filename} (not found)")
            print()
    
    if fixed_count > 0:
        print(f"🎉 Fixed {fixed_count} file(s)!")
    else:
        print("📋 No files needed fixing.")
    
    print("\n=== CSV FORMAT CHECK COMPLETE ===")

if __name__ == "__main__":
    main()