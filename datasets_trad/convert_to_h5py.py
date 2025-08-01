import os
import pandas as pd
import h5py
import argparse
from sklearn.utils import resample

def process_file(file_path, limit_rows=None):
    # Optionally limit rows during file reading
    if limit_rows:
        data = pd.read_csv(file_path, sep='\t', header=None, nrows=limit_rows)
    else:
        data = pd.read_csv(file_path, sep='\t', header=None)
    
    data[1] = data[1].astype(int)  # Ensure query id is integer
    grouped = data.groupby(1)
    processed_data = {}
    
    for name, group in grouped:
        # Resampling logic: downsample if >20, upsample if <20
        if len(group) < 20:
            group = resample(group, replace=True, n_samples=20, random_state=0)
        elif len(group) > 20:
            group = resample(group, replace=False, n_samples=20, random_state=0)
        processed_data[name] = group.values
        
    return processed_data

def convert_to_h5py(original_dir, target_dir, limit_rows=None):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for file_name in os.listdir(original_dir):
        if file_name.endswith('.tsv'):
            file_path = os.path.join(original_dir, file_name)
            processed_data = process_file(file_path, limit_rows)

            h5_file_name = file_name.replace('.tsv', '.h5')
            h5_path = os.path.join(target_dir, h5_file_name)
            
            with h5py.File(h5_path, 'w') as hf:
                for key, value in processed_data.items():
                    hf.create_dataset(str(key), data=value)
            print(f"Converted {file_name} to {h5_file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert tsv files to h5 format.')
    parser.add_argument('--original_dir', required=True, help='Source directory containing tsv files')
    parser.add_argument('--target_dir', required=True, help='Target directory to save h5 files')
    parser.add_argument('--limit_rows', type=int, default=None, 
                        help='Limit processing to first N rows (optional)')
    
    args = parser.parse_args()
    convert_to_h5py(args.original_dir, args.target_dir, args.limit_rows)

