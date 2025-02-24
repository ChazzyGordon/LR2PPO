# Multimodal Label Relevance Ranking via Reinforcement Learning (ECCV2024)
This is an official PyTorch implementation of LR<sup>2</sup>PPO, the ECCV2024 paper is available at [here](https://arxiv.org/abs/2407.13221).
The introduction video can be found at [here](https://www.youtube.com/watch?v=8I3N-XpGBNI).

# Getting Started

## Data Preparation
Download the proposed `LRMovieNet` Dataset from [here](https://huggingface.co/datasets/ChazzyGordon/LRMovieNet).

[Optional]
You can also download `MovieNet` Dataset from its [Official Website](https://movienet.github.io/).



## Initialization Weights Preparation
Download `roberta_base_en_model` and `vit_base_patch16_224_model` weights from [this link](https://drive.google.com/drive/folders/1EVrpImP7f8kSWJW4bQI8OIIXQsHy_1aO) or from its official repositories, and save it in `./pretrained_models/` folder.

## Prerequisites
### Install Requirements
```shell
pip3 install -r requirements.txt
```

### Hardware
* 4 GPUs

# Usage
Before running the following commands, make sure the data path is correct and the GPUs are sufficient (e.g., 4 GPUs).

### Stage 1. Label Relevance Ranking Base Model.
```
sh pointwise.sh <your_stage1>
```

### Stage 2. Reward Model.
```
sh reward_pair_dataloader.sh <your_stage2>
```

### Stage 3. LR<sup>2</sup>PPO.
```
sh ppo.sh <your_stage3>
```

### Evaluation.
```
sh ppo_eval.sh <your_eval> 
```

## Models
We provide logs and checkpoints for the `LRMovieNet` dataset in the `logs/` folder and through [this link](https://drive.google.com/drive/folders/1fRvEuDV-Xji-VHxe01f53VI_9sIcxYOa), respectively.


## License
For more details, please refer to the [LICENSE](./LICENSE) file.

## Acknowledgments
Part of our code is borrowed from the following repositories:
* [TencentPretrain](https://github.com/Tencent/TencentPretrain)
* [PaLM-rlhf-pytorch](https://github.com/lucidrains/PaLM-rlhf-pytorch)

We are grateful for these excellent works and repositories.



## Citation
If you find our work helpful for your research, please consider citing it.
```
@inproceedings{guo2024multimodal,
  title={Multimodal Label Relevance Ranking via Reinforcement Learning},
  author={Guo, Taian and Zhang, Taolin and Wu, Haoqian and Li, Hanjun and Qiao, Ruizhi and Sun, Xing},
  booktitle={European Conference on Computer Vision},
  pages={391--408},
  year={2024},
  organization={Springer}
}
```