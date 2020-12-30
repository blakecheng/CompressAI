import os
from datetime import datetime
import time
import sys
import threading
import yaml
import moxing as mox

def copy_file(obs_path, cache_path):
    if not os.path.exists(os.path.dirname(cache_path)): os.makedirs(os.path.dirname(cache_path))
    print('start copy {} to {}: {}'.format(obs_path, cache_path, datetime.now().strftime("%m-%d-%H-%M-%S")))
    mox.file.copy(obs_path, cache_path)
    print('end copy {} to cache: {}'.format(obs_path, datetime.now().strftime("%m-%d-%H-%M-%S")))


def copy_dataset(obs_path, cache_path):
    if not os.path.exists(cache_path): os.makedirs(cache_path)
    print('start copy {} to {}: {}'.format(obs_path, cache_path, datetime.now().strftime("%m-%d-%H-%M-%S")))
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    mox.file.copy_parallel(obs_path, cache_path)
    print('end copy {} to cache: {}'.format(obs_path, datetime.now().strftime("%m-%d-%H-%M-%S")))


def get_checkpoint(checkpoint_path, s3chekpoint_path):
    def get_time(i):
        return min(600, i)
    start = time.time()
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    i = 1
    while True:
        i = i+1
        print("runtime : {} min ".format((time.time() - start) / 60))
        copy_dataset(checkpoint_path, s3chekpoint_path)

def show_nvidia():
    os.system("nvidia-smi")
    while True:
        time.sleep(1000)
        os.system("nvidia-smi")

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)
        

if __name__ == "__main__":
    path_dict = {
        "default":{
            "s3code_project_path": "s3://bucket-333/chengbin/project/CompressAI",
            "s3data_path":"s3://bucket-333/chengbin/dataset/train_01"
        }
    }

    mode = "developement"
    path_cfg = "default"

    if mode=="developement":
        path_dict[path_cfg]["code_path"]="/home/ma-user/work/CompressAI/code"
        path_dict[path_cfg]["data_path"]="/home/ma-user/work/Hific/data"
        path_dict[path_cfg]["ckpt_path"]="/home/ma-user/work/CompressAI/experiment"
    else:
        path_dict[path_cfg]["code_path"]="/cache/user-job-dir/code"
        path_dict[path_cfg]["data_path"]="/cache/user-job-dir/data"
        path_dict[path_cfg]["ckpt_path"]="/cache/user-job-dir/experiment"

    s3code_path = os.path.join(path_dict[path_cfg]["s3code_project_path"],"code")
    code_path = path_dict[path_cfg]["code_path"]

    s3data_path = path_dict[path_cfg]["s3data_path"]
    data_path = path_dict[path_cfg]["data_path"]

    copy_dataset(s3code_path, code_path)
    # copy_dataset(s3data_path, data_path)

    sys.path.insert(0, code_path)  # "home/work/user-job-dir/" + leaf folder of src code
    os.chdir(code_path)
    os.system("pwd")

    checkpoint_path = path_dict[path_cfg]["ckpt_path"]
    s3savepath = os.path.join(path_dict[path_cfg]["s3code_project_path"],"experiments")

    t = threading.Thread(target=get_checkpoint, args=(checkpoint_path,s3savepath,))
    t.start()

    t = threading.Thread(target=show_nvidia)
    t.start()

    os.system("pwd")

    if mode=="developement":
        os.system("python examples/train_scale.p -d ../../Hific/data --model scale_bmshj2018_factorized --loss ID")
    else:
        os.system("pip install -r requirements.txt")
        os.system("pip install -e .")
        os.system("python examples/train_scale.py -d %s --model scale_bmshj2018_factorized --scale 4"%(data_path))
# os.system("scp save_models/alexnet-owt-4df8aa71.pth /home/ma-user/.cache/torch/hub/")


