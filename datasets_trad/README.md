# LTR Transfer Learning Benchmark Preparation Guide

This document details the construction process of our cross-domain learning-to-rank (LTR) benchmark using MSLR-WEB10K and MQ2008 datasets. The benchmark enables rigorous evaluation of transfer learning effectiveness while preventing data leakage through careful dataset partitioning and feature space alignment.

## Motivation and Rationale

### Why These Datasets?
- **Complementary Characteristics**: WEB10K (136-dim features) and MQ2008 (46-dim features) represent distinct feature spaces from different search domains
- **Established Baselines**: Both are widely-used LTR benchmarks with standardized evaluation protocols
- **Real-World Relevance**: Derived from commercial search engines, ensuring practical significance

### Key Design Principles:
1. **Strict Domain Separation**  
   - **Source Domain**: WEB10K (136-dim features)
   - **Target Domain**: MQ2008 (46-dim features)

2. **Feature Space Unification**  
   - Align heterogeneous features via MLP projections (136/46-dim $\rightarrow$ 768-dim)
   - Lightweight mapping model (i.e., 2-layer MLP) trained on Fold2 data from both domains
   - Fold1 data preserved for transfer experiments

3. **Leakage-Free Guarantee**  
   - Query ID offset (100k+) prevents domain overlap
   - Zero document/content intersection
   - Test data (MQ2008-Fold1) never exposed during feature alignment

## Dataset Preparation Pipeline

### 1. Dataset Acquisition & Structure

Download datasets from official sources:
- **MSLR-WEB10K**: [Official Website](https://www.microsoft.com/en-us/research/project/mslr/)
- **MQ2008**: [Official Website](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/letor-4-0/)

Organize datasets in the following directory structure:
```text
trad_datasets/
├── MSLR-WEB10K/       # Source domain data
│   └── Fold1/         # Transfer learning (136-dim)
│   └── Fold2/         # Mapping training (136-dim)
└── MQ2008/            # Target domain data
    └── Fold1/         # Transfer learning (46-dim)
    └── Fold2/         # Mapping training (46-dim)
```

### 2. Data Preprocessing

Execute format conversion and quality checks:
```bash
python3 preprocess_data.py \
    --input_dir trad_datasets/MSLR-WEB10K/Fold1/ \
    --output_dir trad_datasets/preprocessed/MSLR-WEB10K/Fold1/
python3 preprocess_data.py \
    --input_dir trad_datasets/MSLR-WEB10K/Fold2/ \
    --output_dir trad_datasets/preprocessed/MSLR-WEB10K/Fold2/
python3 preprocess_data.py \
    --input_dir trad_datasets/MQ2008/Fold1/ \
    --output_dir trad_datasets/preprocessed/MQ2008/Fold1/
python3 preprocess_data.py \
    --input_dir trad_datasets/MQ2008/Fold2/ \
    --output_dir trad_datasets/preprocessed/MQ2008/Fold2/
```

### 3. Query ID Disjointing

Ensure no query ID overlap between domains:
```bash

# WEB10K & MQ2008
python3 check_intersec.py --mslr_path trad_datasets/preprocessed/MSLR-WEB10K/Fold1 --mq2008_path trad_datasets/preprocessed/MQ2008/Fold1
python3 check_intersec.py --mslr_path trad_datasets/preprocessed/MSLR-WEB10K/Fold2 --mq2008_path trad_datasets/preprocessed/MQ2008/Fold2

# MQ2008
python3 make_indices_disjoint.py \
    --mq2008_path trad_datasets/preprocessed/MQ2008/Fold1 \
    --dst_path trad_datasets/preprocessed/MQ2008/Fold1_qid10w/ \
    --offset 100000
python3 make_indices_disjoint.py \
    --mq2008_path trad_datasets/preprocessed/MQ2008/Fold2 \
    --dst_path trad_datasets/preprocessed/MQ2008/Fold2_qid10w/ \
    --offset 100000
```

### 4. HDF5 Conversion for Original Fold2 Features

Optimize storage for large-scale feature processing:
```bash
# WEB10K Fold2 (First 50k lines)
python3 convert_to_h5py.py \
    --original_dir trad_datasets/preprocessed/MSLR-WEB10K/Fold2/ \
    --target_dir trad_datasets/h5py_data/MSLR-WEB10K/Fold2_5w/ \
    --limit_rows 50000

# MQ2008 Fold2 
python3 convert_to_h5py.py \
    --original_dir trad_datasets/preprocessed/MQ2008/Fold2_qid10w/ \
    --target_dir trad_datasets/h5py_data/MQ2008/Fold2_qid10w/ 
```

### 5. Cross-Domain Feature Mapping

Train lightweight mapping model (i.e., 2-layer MLP) on Fold2 HDF5 data from both domains:

```bash
sh pointwise_2data_trad.sh Web10kF2_5w_mq2008_2data_s1
```

**Trained checkpoint available**: [Google Drive](https://drive.google.com/drive/folders/1SNE0dYtWzqpvXkxXHBaSXzr8xqlEfn8B)


### 6. Feature Projection

Generate unified 768-dim representations using the lightweight mapping model:
```bash
# WEB10K
sh pointwise_2data_infer_trad.sh pointwise_2data_infer_trad_web10k Web10kF2_5w_mq2008_2data_s1 \
    datasets_trad/trad_datasets/preprocessed/MSLR-WEB10K/Fold1 \
    datasets_trad/trad_datasets/preprocessed/MSLR-WEB10K/Fold1_dim768_F2ckpt

# MQ2008
sh pointwise_2data_infer_trad.sh pointwise_2data_infer_trad_mq2008 Web10kF2_5w_mq2008_2data_s1 \
    datasets_trad/trad_datasets/preprocessed/MQ2008/Fold1_qid10w \
    datasets_trad/trad_datasets/preprocessed/MQ2008/Fold1_qid10w_dim768_F2ckpt
```

### 7. HDF5 Conversion for Aligned Fold1 Features

Optimize storage for large-scale feature processing:
```bash
# WEB10K Fold1
python3 convert_to_h5py.py \
    --original_dir trad_datasets/preprocessed/MSLR-WEB10K/Fold1_dim768_F2ckpt/ \
    --target_dir trad_datasets/h5py_data/MSLR-WEB10K/Fold1_dim768_F2ckpt/

# MQ2008 Fold1
python3 convert_to_h5py.py \
    --original_dir trad_datasets/preprocessed/MQ2008/Fold1_qid10w_dim768_F2ckpt/ \
    --target_dir trad_datasets/h5py_data/MQ2008/Fold1_qid10w_dim768_F2ckpt 
```

### 8. Merging WEB10K and MQ2008 Fold1 Training Data

Execute the script to combine WEB10K and MQ2008 Fold1 datasets (with aligned 768-dim features) into a unified training set for transfer learning experiments:
```bash
sh combine_web10k_mq2008_fold1.sh \
    trad_datasets/preprocessed/MSLR-WEB10K/Fold1_dim768_F2ckpt \
    trad_datasets/preprocessed/MQ2008/Fold1_qid10w_dim768_F2ckpt \
    trad_datasets/preprocessed/WEB10K_MQ2008/Fold1_qid10w_dim768_F2ckpt
```

### 9. HDF5 Conversion for Merged Fold1 Features

Convert the combined dataset into HDF5 format for efficient storage and model training:
```bash
python3 convert_to_h5py.py \
    --original_dir trad_datasets/preprocessed/WEB10K_MQ2008/Fold1_qid10w_dim768_F2ckpt/ \
    --target_dir trad_datasets/h5py_data/WEB10K_MQ2008/Fold1_qid10w_dim768_F2ckpt/
```

### 10. Final Data Structure
After completing all steps, you'll obtain:
```text
trad_datasets/h5py_data/
├── MSLR-WEB10K/Fold1_dim768_F2ckpt/             # Source domain features (768-dim)
└── MQ2008/Fold1_qid10w_dim768_F2ckpt/           # Target domain features (768-dim)
└── WEB10K_MQ2008/Fold1_qid10w_dim768_F2ckpt/    # Combined features (768-dim)
```

These prepared features are now ready for the three-stage transfer learning pipeline.
