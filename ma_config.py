import os
import argparse
import subprocess
import sys
import traceback
import pathlib
import subprocess
import moxing as mox
from pathlib import Path
mox.file.shift('os','mox')
from concurrent.futures import ProcessPoolExecutor

class MAEnv:
    def __init__(self, ma_config):
        self.ma_config = ma_config
        self.ma_config.nnodes = int(os.environ.get('MA_NUM_HOSTS', 1))

        self.host = 'localhost'
        self.port = self.ma_config.master_port

    def _setup_distributed_env(self):
        try:
            if self.ma_config.nnodes > 1:
                self.ma_config.node_rank = int(os.environ['VC_TASK_INDEX']) \
                    if os.environ.get('VC_TASK_INDEX') is not None else int(os.environ.get('MA_TASK_INDEX'))

            else:
                # self.ma_config.node_rank=0
                assert self.ma_config.node_rank == 0

            rdzv_endpoint = '{}-{}-0.{}:{}'.format(os.environ['MA_VJ_NAME'],
                                                   os.environ['MA_TASK_NAME'],
                                                   os.environ['MA_VJ_NAME'], self.ma_config.master_port)

            self.host, self.port = rdzv_endpoint[:-5], rdzv_endpoint[-4:]

        except:
            self.ma_config.nnodes = 1
            self.ma_config.node_rank = 0
            self.host = "localhost"
            self.port = self.ma_config.master_port
            traceback.print_exc()

    def setup_ma_env(self):
        self._setup_distributed_env()
        if self.ma_config.nnodes > 1:
            self.num_gpus = 8
        else:
            import torch

            self.num_gpus = torch.cuda.device_count()
        self.nnodes=self.ma_config.nnodes
        self.node_rank=self.ma_config.node_rank

    def prepare_env(self):

        os.system('pip install --upgrade pip')
        os.system('pip install -e .')
        os.system('pip install -e ".[train]"')
        
        
        # os.system('pip uninstall deepspeed -y')
        
        # os.chdir('./wheels')
        # os.system('DS_BUILD_FUSED_ADAM=1 DS_BUILD_CPU_ADAM=1 pip install deepspeed==0.9.5')
        # os.system('pip install referencing-0.30.2-py3-none-any.whl')
        # os.system('pip install jsonschema_specifications-2023.7.1-py3-none-any.whl')
        # os.system('pip install jsonschema-4.19.0-py3-none-any.whl')
        
        # os.system('pip install joblib-1.3.2-py3-none-any.whl')
        # os.system('pip install einops_exts-0.0.4-py3-none-any.whl')
        # os.system('pip install aiofiles-23.2.1-py3-none-any.whl')
        # os.system('pip install async_timeout-4.0.3-py3-none-any.whl')
        # os.system('pip install bitsandbytes-0.41.0-py3-none-any.whl')
        # os.system('pip install gradio_client-0.2.9-py3-none-any.whl')
        
        # os.system('pip install scikit_learn-1.2.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl')
        
        # os.chdir('..')
        
        
        # os.system('pip install xformers')
        # os.system('pip install -e .')
        # os.system('pip install -e ".[train]"')
        





