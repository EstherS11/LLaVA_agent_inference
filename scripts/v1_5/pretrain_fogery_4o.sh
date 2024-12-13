#!/bin/bash
#    
deepspeed llava/train/train_xformers.py \
    --mm_projector_lr 1e-3 \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path /cache/llava_forgery_SFT_0726 \
    --version v1 \
    --data_path /home/ma-user/work/LLaVA/data/sft_forgery_data/qa_pairs_sft_casiav2_merge_coverage.json \
    --image_folder /home/ma-user/work/LLaVA/data/sft_forgery_data \
    --vision_tower /home/ma-user/work/llava_ckpt/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_mm_mlp_adapter True \
    --image_aspect_ratio no \
    --group_by_modality_length True \
    --bf16 False \
    --output_dir /cache/checkpoints/llava-v1.5-7b-fogery-pretrain-4o \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to tensorboard 
    # --fp16 True
