TRAIN_PATH=LRMovieNet/first_stage_data.json
DEV_PATH=LRMovieNet/val_data.json
TEST_PATH=LRMovieNet/test_data.json



OUTPUT_MODEL_DIR=pointwise_ckpt/$1
mkdir -p ${OUTPUT_MODEL_DIR}

LOG_DIR=pointwise_logs/$1
mkdir -p ${LOG_DIR}

train_args=(
    --train_path $TRAIN_PATH 
    --dev_path $DEV_PATH 
    --test_path $TEST_PATH 
    --epochs_num 15
    --mask fully_visible 
    --output_model_path ${OUTPUT_MODEL_DIR}/finetuned_model.bin
    --log_path ${LOG_DIR}/$1.txt
    --exp_name $1
    --batch_size 2
    --seq_length 196
    --visual_feat_dim 768
    --max_imgs 16
    --report_steps 150
    --mode reg
    --max_tags 20
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

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 29570 finetune/pointwise.py \
                                   "${train_args[@]}" \
                                   "${text_args[@]}" \
                                   "${vit_args[@]}"