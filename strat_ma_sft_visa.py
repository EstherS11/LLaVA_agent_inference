# -*- coding: utf-8 -*-
import os
import moxing as mox
import argparse
import time
import sys

from ma_config import MAEnv

def prepare_model():
    mox.file.copy_parallel('s3://doclm/zhaojie/interns/zhangfanrui/llava_modelarts/model/', '/home/ma-user/work/zhangfanrui/model/')
    print('model 准备完成')

def prepare_data():
    # 先 mox 再解压
    mox.file.copy_parallel('s3://doclm/zhaojie/interns/zhangfanrui/dataset/VisA.zip', '/home/ma-user/work/zhangfanrui/data/VisA.zip')
                                                                                                
    os.system('unzip -nq /home/ma-user/work/zhangfanrui/data/VisA.zip -d /home/ma-user/work/zhangfanrui/data/')
    print('data 准备完成')

## mox.file.copy_parallel('s3://doclm/zhaojie/interns/zhangfanrui/llava_modelarts/data/json', '/home/ma-user/work/zhangfanrui/data/json')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--deepspeed', type=str, default='./scripts/zero2_offload.json', help='the input data path')
    parser.add_argument('--train_url', type=str, default='/cache/sft_bbox/', help='the input data path')
    parser.add_argument('--model_name_or_path', type=str, default='/home/ma-user/work/zhangfanrui/model/llava-v1.5-7b', help='the input data path')
    parser.add_argument('--version', type=str, default='v1', help='the input data path')
    parser.add_argument('--image_folder', type=str, default='/home/ma-user/work/zhangfanrui/data/VisA', help='the input data path')
    parser.add_argument('--data_path', type=str, default='/home/ma-user/work/zhangfanrui/data/VisA/qa_pairs_anomaly_visa.json', help='the input data path')
    parser.add_argument('--vision_tower', type=str, default='/home/ma-user/work/zhangfanrui/model/clip-vit-large-patch14-336', help='the input data path')
    ### 选择训的比较好的adapter
    parser.add_argument('--pretrain_mm_mlp_adapter', type=str, default='/cache/pretrain_0615/mm_projector.bin', help='the input data path')
    parser.add_argument('--pretrain_mask_mlp_adapter', type=str, default='/cache/pretrain_0615/mask_projector.bin', help='the input data path')
    parser.add_argument('--mm_projector_type', type=str, default='mlp2x_gelu', help='the input data path')
    parser.add_argument('--mm_vision_select_layer', type=int, default=-2, help='world size')
    parser.add_argument('--output_dir', type=str, default='', help='the input data path')
    parser.add_argument('--image_aspect_ratio', type=str, default='pad', help='the input data path')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='world size')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help='world size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help='world size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='world size')
    parser.add_argument('--save_steps', type=int, default=2000, help='world size')
    parser.add_argument('--save_total_limit', type=int, default=1, help='world size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='world size')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='world size')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='world size')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine', help='the input data path')
    parser.add_argument('--logging_steps', type=int, default=1, help='the input data path')
    parser.add_argument('--tf32', action="store_true", help='the input data path')
    parser.add_argument('--model_max_length', type=int, default=2048, help='the input data path')
    parser.add_argument('--gradient_checkpointing', action="store_true", help='the input data path')
    parser.add_argument('--dataloader_num_workers', type=int, default=4, help='the input data path')
    parser.add_argument('--lazy_preprocess', action="store_true", help='the input data path')
    parser.add_argument('--fp16', action="store_true", help='the input data path') 
    parser.add_argument('--debug', action="store_true", help='the input data path')
    parser.add_argument('--lora_enable', action="store_true", help='node rank')
    parser.add_argument('--freeze_backbone', action="store_true", help='node rank')
    parser.add_argument('--lora_r', type=int, default=0, help='world size')
    parser.add_argument('--lora_alpha', type=int, default=0, help='world size')
    parser.add_argument('--mm_projector_lr', type=float, default=1e-5, help='world size')
    parser.add_argument('--master_port', type=str, default='8524', help='the input data path')
    args, unparsed = parser.parse_known_args()

    # os.system('nvcc --version')


    # update llava code in workdir
    # test commit
    # mox.file.copy_parallel('s3://doclm/zhaojie/interns/zhangfanrui/llava_modelarts/code/LLaVA/','/home/ma-user/modelarts/user-job-dir/LLaVA')

    # backup run code
    # mox.file.copy_parallel('/home/ma-user/modelarts/user-job-dir/LLaVA',os.path.join(args.train_url,'code/LLaVA'))

    # prepare model 把 clip 和 intern 两个模型 mox 到开发环境中
    # prepare_model()

    # prepare data 主要是 mox 图片的压缩包，并解压在开发环境中
    prepare_data()

    maenv = MAEnv(args)
    # os.system('which python')
    maenv.setup_ma_env()
    # os.system('pip list')
    # maenv.prepare_env()
    # os.system('pip list')
##           
    cmd = f"torchrun --nnodes={maenv.nnodes} --nproc_per_node={maenv.num_gpus} \
          --node_rank={maenv.node_rank} --master_addr={maenv.host} --master_port={maenv.port} \
          llava/train/train_xformers.py \
          --deepspeed {args.deepspeed} \
          --pretrain_mask_mlp_adapter {args.pretrain_mask_mlp_adapter} \
          --pretrain_mm_mlp_adapter {args.pretrain_mm_mlp_adapter} \
          --model_name_or_path {args.model_name_or_path} \
          --version {args.version} \
          --data_path {args.data_path} \
          --image_folder {args.image_folder} \
          --vision_tower {args.vision_tower} \
          --mm_projector_type {args.mm_projector_type} \
          --tune_mm_mlp_adapter {False} \
          --mm_vision_select_layer {args.mm_vision_select_layer} \
          --mm_use_im_start_end {False} \
          --mm_use_im_patch_token {False} \
          --bf16 {False} \
          --output_dir {args.train_url} \
          --num_train_epochs {args.num_train_epochs} \
          --per_device_train_batch_size {args.per_device_train_batch_size} \
          --per_device_eval_batch_size {args.per_device_eval_batch_size} \
          --gradient_accumulation_steps {args.gradient_accumulation_steps} \
          --evaluation_strategy no \
          --image_aspect_ratio pad \
          --group_by_modality_length True \
          --save_strategy steps \
          --save_steps {args.save_steps} \
          --save_total_limit {args.save_total_limit} \
          --learning_rate {args.learning_rate} \
          --weight_decay {args.weight_decay} \
          --warmup_ratio {args.warmup_ratio} \
          --lr_scheduler_type {args.lr_scheduler_type} \
          --logging_steps {args.logging_steps} \
          --tf32 {args.tf32} \
          --model_max_length {args.model_max_length} \
          --gradient_checkpointing {True} \
          --dataloader_num_workers {args.dataloader_num_workers} \
          --lazy_preprocess {True} \
          --fp16 {args.fp16} \
          --report_to tensorboard " 
    if args.lora_enable:
        cmd += " --lora_enable {} --lora_r {} --lora_alpha {} --mm_projector_lr {} ".format(args.lora_enable,args.lora_r,args.lora_alpha,args.mm_projector_lr)
    # if len(args.pretrain_mm_mlp_adapter)>1:
    #     cmd+= f'--pretrain_mm_mlp_adapter {args.pretrain_mm_mlp_adapter} '

    print('=====>run cmd:{}'.format(cmd), flush=True)
    os.system(cmd)


    if args.debug:
        while True:
            time.sleep(60)
