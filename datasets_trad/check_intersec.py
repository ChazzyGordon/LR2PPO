import pandas as pd
import os
import sys
import argparse

# Verify query ID overlap between MSLR-WEB10K and MQ2008 datasets
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mslr_path', default='trad_datasets/preprocessed/MSLR-WEB10K/Fold1', help='Path to MSLR-WEB30K dataset')
    parser.add_argument('--mq2008_path', default='trad_datasets/preprocessed/MQ2008/Fold1', help='Path to MQ2008 dataset')
    args = parser.parse_args()

    # Read MSLR-WEB10K data
    df1 = pd.read_csv(os.path.join(args.mslr_path, 'train.tsv'), sep='\t', header=None)
    df2 = pd.read_csv(os.path.join(args.mslr_path, 'test.tsv'), sep='\t', header=None)
    
    # Read MQ2008 data
    df3 = pd.read_csv(os.path.join(args.mq2008_path, 'train.tsv'), sep='\t', header=None)
    df4 = pd.read_csv(os.path.join(args.mq2008_path, 'test.tsv'), sep='\t', header=None)
    
    # Extract query IDs and create sets
    set_A = set(df1[1].tolist() + df2[1].tolist())  # MSLR IDs
    set_B = set(df3[1].tolist() + df4[1].tolist())  # MQ2008 IDs
    
    # Find intersection
    intersection = set_A & set_B
    
    # Output analysis results
    print(f"Intersection size: {len(intersection)}")
    print(f"Min(A): {min(set_A)}, Max(A): {max(set_A)}")
    print(f"Min(B): {min(set_B)}, Max(B): {max(set_B)}")
    
    # Observed ranges: 
    #   A: 1.0 - 31531.0
    #   B: 10002.0 - 19997.0