#!/bin/bash
    # --pretrain_mask_mlp_adapter /data/zfr/llava_modelarts/out/llava-v1.5-7b-pretrain_pub/mask_projector.bin \
    # --pretrain_mm_mlp_adapter /data/zfr/llava_modelarts/out/llava-v1.5-7b-pretrain_pub/mm_projector.bin \
deepspeed llava/train/train_xformers.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-5 \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path /home/ma-user/work/llava_ckpt/pretrain/llava-v1.5-7b \
    --version v1 \
    --data_path /home/ma-user/work/LLaVA/data/sft_forgery_data/qa_pairs_forgery_sft_bbox_updated.json \
    --image_folder /home/ma-user/work/LLaVA/data/sft_forgery_data \
    --vision_tower /home/ma-user/work/llava_ckpt/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --output_dir ./checkpoints/llava-v1.5-7b-fogery-lora5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --fp16 True
