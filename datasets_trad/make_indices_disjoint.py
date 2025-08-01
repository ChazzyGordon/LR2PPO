import os
import pandas as pd
import argparse

# Command-line argument parser
parser = argparse.ArgumentParser(description='Make MQ2008 query IDs disjoint by adding a unique offset')
parser.add_argument('--mq2008_path', required=True, 
                    help='Base path to the original MQ2008 dataset fold')
parser.add_argument('--dst_path', required=True,
                    help='Output directory path for processed files')
parser.add_argument('--offset', type=int, required=True,
                    help='Offset value to add to all query IDs for uniqueness')
args = parser.parse_args()

# Configure paths
src_dir = args.mq2008_path  # Input directory from command line
dst_dir = args.dst_path     # Output directory from command line
offset = args.offset        # Offset value for query IDs

# Create output directory if needed
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
    print(f"Created output directory: {dst_dir}")

# Process each TSV file in the source directory
for filename in os.listdir(src_dir):
    if filename.endswith('.tsv'):
        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(dst_dir, filename)
        
        # Read TSV file into DataFrame
        df = pd.read_csv(src_file, sep='\t', header=None)
        
        # Add offset to all query IDs in column 1
        df[1] = df[1].astype(int) + offset
        
        # Save modified file
        df.to_csv(dst_file, sep='\t', header=None, index=False)
        print(f"Processed: {src_file} -> {dst_file}")

# Note: Using 100000 offset ensures no QID overlap with MSLR-WEB10K (max QID ~31531)