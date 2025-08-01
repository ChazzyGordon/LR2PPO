GPUS=0
# GPUS=0,1
# GPUS=0,1,2,3

NGPUS=1
# NGPUS=2
# NGPUS=4

REPORT_STEPS=10
PORT=29591

OUTPUT_MODEL_DIR=pointwise_ckpt_trad/$1
mkdir -p ${OUTPUT_MODEL_DIR}

LOG_DIR=pointwise_logs_trad/$1
mkdir -p ${LOG_DIR}

DIM_PROJ_CKPT_PATH=pointwise_ckpt_trad/$2/finetuned_model.bin
INPUT_DIR=$3
OUTPUT_DIR=$4

train_args=(
    --epochs_num 15
    --mask fully_visible 
    --output_model_path ${OUTPUT_MODEL_DIR}/finetuned_model.bin
    --log_path ${LOG_DIR}/$1.txt
    --exp_name $1
    --batch_size 2
    --seq_length 196
    --visual_feat_dim 768
    --max_imgs 16
    --report_steps $REPORT_STEPS
    --mode reg
    --max_tags 20
    --dim_proj_ckpt_path ${DIM_PROJ_CKPT_PATH}
    --input_dir ${INPUT_DIR}
    --output_dir ${OUTPUT_DIR}
)

text_args=(
    --pretrained_model_path pretrained_models/roberta_base_en_model.bin
    --vocab_path models/huggingface_gpt2_vocab.txt 
    --merges_path models/huggingface_gpt2_merges.txt
    --tokenizer bpe
    --config_path models/xlm-roberta/base_config.json
    --encoder transformer 
)

vit_args=(
    --vit_pretrained_model_path pretrained_models/vit_base_patch16_224_model.bin
    --vit_tokenizer virtual
    --vit_config_path models/vit/base-16-224_config.json
    --vit_encoder transformer
)

CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NGPUS --master_port $PORT finetune/pointwise_2data_infer_trad.py \
                                   "${train_args[@]}" \
                                   "${text_args[@]}" \
                                   "${vit_args[@]}"