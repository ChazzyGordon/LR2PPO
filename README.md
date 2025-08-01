# Multimodal Label Relevance Ranking via Reinforcement Learning (ECCV2024)
This is the official PyTorch implementation of LR<sup>2</sup>PPO. The ECCV2024 paper is available at [arXiv](https://arxiv.org/abs/2407.13221).  
Introduction video: [YouTube](https://www.youtube.com/watch?v=8I3N-XpGBNI)

## Getting Started
### Data Preparation
#### For LRMovieNet Benchmark
- Download dataset: [HuggingFace Hub](https://huggingface.co/datasets/ChazzyGordon/LRMovieNet)  
- *Optional*: Original MovieNet dataset [Official Website](https://movienet.github.io/)

#### For MSLR-Web10K → MQ2008 Transfer Task
- Pre-processed datasets (`datasets_trad`) available: [Google Drive](https://drive.google.com/drive/folders/18rU9fORPvQNdBMd1rfZa-OnZbWn7dSSF)  
- *Optional preparation*:  
  - Follow dataset generation guide: [`datasets_trad/README_TRADDATA.md`](./datasets_trad/README_TRADDATA.md)  
  - Access source datasets:  
    • MSLR-Web10K: [Microsoft Research](https://www.microsoft.com/en-us/research/project/mslr/)  
    • MQ2008: [LETOR 4.0](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/letor-4-0/)  

### Initialization Weights
Download required weights for **both benchmarks**:  
- `roberta_base_en_model` and `vit_base_patch16_224_model`  
- Source: from [Google Drive](https://drive.google.com/drive/folders/1EVrpImP7f8kSWJW4bQI8OIIXQsHy_1aO) or from its official repositories  
- Save in: `./pretrained_models/`

### Prerequisites
```shell
pip3 install -r requirements.txt
```
**Hardware Requirement**: 4 GPUs

## Usage Instructions
### For LRMovieNet Benchmark
```shell
# Stage 1: Base Model
sh pointwise.sh <your_stage1>

# Stage 2: Reward Model
sh reward_pair_dataloader.sh <your_stage2>

# Stage 3: LR<sup>2</sup>PPO
sh ppo.sh <your_stage3>

# Evaluation
sh ppo_eval.sh <your_eval>
```

### For MSLR-Web10K → MQ2008 Transfer Task
```shell
# Stage 1: Base Model
sh pointwise_trad.sh <your_stage1>

# Stage 2: Reward Model
sh reward_trad.sh <your_stage2>

# Stage 3: LR<sup>2</sup>PPO
sh ppo_trad.sh <your_stage3>

# Evaluation
sh ppo_eval_trad.sh <your_eval>
```

## Model Checkpoints
### LRMovieNet Benchmark
- Download: [Google Drive](https://drive.google.com/drive/folders/1fRvEuDV-Xji-VHxe01f53VI_9sIcxYOa)

### MSLR-Web10K → MQ2008 Transfer
- Download: [Google Drive](https://drive.google.com/drive/folders/1OGtzokoqmeow13NtN6KrwdHaDyrurNW0)

## License
See [LICENSE](./LICENSE) for details.

## Acknowledgments
Code components borrowed from:  
- [TencentPretrain](https://github.com/Tencent/TencentPretrain)  
- [PaLM-rlhf-pytorch](https://github.com/lucidrains/PaLM-rlhf-pytorch)  
- [benchmarks](https://github.com/catboost/benchmarks) (Transfer Task)

We are grateful for these excellent works and repositories.

## Citation
If you found our work helpful in your research, please consider citing it.
```bibtex
@inproceedings{guo2024multimodal,
  title={Multimodal Label Relevance Ranking via Reinforcement Learning},
  author={Guo, Taian and Zhang, Taolin and Wu, Haoqian and Li, Hanjun and Qiao, Ruizhi and Sun, Xing},
  booktitle={European Conference on Computer Vision},
  pages={391--408},
  year={2024},
  organization={Springer}
}
```