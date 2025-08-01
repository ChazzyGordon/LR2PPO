import os
import argparse
from preprocess import msrank, mq2008  # https://github.com/catboost/benchmarks/blob/master/ranking/preprocess.py

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Data preprocessing script')
    parser.add_argument('--input_dir', type=str, default='trad_datasets/MQ2008/Fold2/',
                        help='Input directory containing raw datasets')
    parser.add_argument('--output_dir', type=str, default='trad_datasets/preprocessed/MQ2008/Fold2/',
                        help='Output directory for processed data')
    args = parser.parse_args()
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Identify the dataset type and call the corresponding processing function
    input_dir = args.input_dir.rstrip('/')
    
    # Check if it is the MSLR-WEB dataset
    if "MSLR-WEB" in input_dir:

        print(f"Processing MSLR-WEB dataset from: {input_dir}")
        msrank(input_dir, args.output_dir)
    
    # Check if it is the MQ2008 dataset
    elif "MQ2008" in input_dir:

        print(f"Processing MQ2008 dataset from: {input_dir}")
        mq2008(input_dir, args.output_dir)
    
    else:
        raise ValueError(f"Unsupported dataset type in directory: {input_dir}")
