# -*- coding: utf-8 -*-
import os
import moxing as mox
import argparse
import time
import sys

from ma_config import MAEnv

def prepare_model():
    mox.file.copy_parallel('s3://doclm/zhaojie/interns/zengyu/llava/model/', '/home/ma-user/work/zengyu/model/')
    print('model 准备完成')

def prepare_data():
    # 先 mox 再解压
    mox.file.copy_parallel('s3://doclm/zhaojie/interns/zengyu/llava/data/', '/home/ma-user/work/zengyu/data/')

    os.system('unzip -nq /home/ma-user/work/zengyu/data/image/coco/train2017.zip -d /home/ma-user/work/zengyu/data/image/coco/')
    print('data 准备完成')

os.system('nvcc --version')


# update llava code in workdir
# test commit
mox.file.copy_parallel('s3://doclm/zhaojie/interns/zengyu/llava/code/LLaVA/','/home/ma-user/modelarts/user-job-dir/LLaVA')

# backup run code
mox.file.copy_parallel('/home/ma-user/modelarts/user-job-dir/LLaVA',os.path.join(args.train_url,'code/LLaVA'))

# prepare model 把 clip 和 intern 两个模型 mox 到开发环境中
prepare_model()

# prepare data 主要是 mox 图片的压缩包，并解压在开发环境中
prepare_data()


while True:
    time.sleep(60)