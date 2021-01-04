# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.layers import GDN
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv
from torch.utils.tensorboard import SummaryWriter
from compressai.utils import setup_generic_signature
from torch.utils.tensorboard import SummaryWriter
from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor)
from tqdm import tqdm 
import os
import logging
from compressai.models import Scale_FactorizedPrior,MultiScale_FactorizedPrior,MultiEB_Scale_FactorizedPrior
import getopt


class AutoEncoder(CompressionModel):
    """Simple autoencoder with a factorized prior """

    def __init__(self, N=128):
        super().__init__(entropy_bottleneck_channels=N)

        self.encode = nn.Sequential(
            conv(3, N, kernel_size=9, stride=4),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
        )

        self.decode = nn.Sequential(
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=9, stride=4),
        )

    def forward(self, x):
        y = self.encode(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decode(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }





class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out

class WeightedRateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2,scale=8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.scale = scale
        self.weight = [2*i for i in range(scale)]

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        
        out["mse_loss"] = sum(
            (self.weight[i]*self.mse(output["rgb_list"][i],target))
            for i in range(len(output["rgb_list"]))
        )
        # out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out

class InformationDistillationRateDistortionLoss(nn.Module):
    def __init__(self,lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
    
    def forward(self,output,target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = 0
        for st,te in zip(output["rgb_list"],output["rgb_list"][1:]+[target]):
            out["mse_loss"] += self.mse(st, te)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        return out
        



class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    def __init__(self,log_interval=10,test_inteval=100,save_dirs=None,max_iter=1e6):
        self.iteration = 0
        self.log_interval = log_interval
        self.test_inteval = test_inteval
        self.best_loss = 1000
        self.save_dirs = save_dirs
        self.max_iter = int(max_iter)


    def updata(self):
        self.iteration+=1
        if self.iteration%self.log_interval==0:
            return True
        return False

# class PSNR(nn.Module):
#     def __init__(self, max_val):
#         super(PSNR, self).__init__()

#         base10 = torch.log(torch.tensor(10.0))
#         max_val = torch.tensor(max_val).float()

#         self.register_buffer('base10', base10)
#         self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

#     def __call__(self, a, b):
#         mse = torch.mean((a.float() - b.float()) ** 2)
    
#         if mse == 0:
#             return 0

#         return self.max_val - 10 * torch.log(mse) / self.base10

def compute_psnr(a,b,max_val=torch.tensor(255.0)):
    mse = torch.mean((a.float() - b.float()) ** 2)
    if mse == 0:
        mse=torch.tensor([1e-6]).cuda()
    
    return (20 * torch.log(max_val) - 10 * torch.log(mse))/torch.log(torch.tensor(10.0))



def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm,test_dataloader,args,writer=None,test_writer=None,logger=None
):
    model.train()
    device = next(model.parameters()).device

    pbar = tqdm(train_dataloader,ncols=160)
    for i, d in enumerate(pbar):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if logger.updata():
            if writer is not None:
                step = logger.iteration
                writer.add_scalar("mse_loss",out_criterion["mse_loss"], step)
                writer.add_scalar("bpp_loss",out_criterion["bpp_loss"], step)
                writer.add_scalar("loss",out_criterion["loss"], step)
                writer.add_scalar("aux_loss",aux_loss, step),
                
        pbar.set_description(
        "Train epoch {}: Loss: {:.3f} | MSE loss:{:.3f} | Bpp loss: {:.2f} | Aux loss: {:.2f}".format(epoch,
            out_criterion["loss"].item(),
            out_criterion["mse_loss"].item(),
            out_criterion["bpp_loss"].item(),
            aux_loss.item()
        )
        )
        pbar.set_postfix(iterations="{}/{}".format(logger.iteration,logger.max_iter))

        if logger.iteration%logger.test_inteval==0:
            print("\ntesting:")
            step = logger.iteration
            writer.add_images('gen_recon', torch.cat((out_net["x_hat"][:4],d[:4]),dim=0), step)
            loss = test(logger.iteration, test_dataloader, model, criterion,test_writer=test_writer,logger=logger)
            is_best = loss < logger.best_loss
            logger.best_loss = min(loss, logger.best_loss)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "iteration": logger.iteration,
                    "state_dict": model.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "args":args
                },
                is_best, path= logger.save_dirs["checkpoints_save"]
            )
            model.train()
        



def test(iterations, test_dataloader, model, criterion,test_writer=None,logger=None):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr_list = []
    for i in range(model.scale):
        psnr_list.append(AverageMeter())

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            out_net["x_hat"] = torch.clamp(out_net["x_hat"],min=0.,max=1.)
            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            for i,rgb in enumerate(out_net["rgb_list"]):
                psnr_list[i].update(compute_psnr(rgb*255.0,d*255.0))

        if test_writer is not None:
            step = logger.iteration
            test_writer.add_scalar("mse_loss",mse_loss.avg, step)
            test_writer.add_scalar("bpp_loss",bpp_loss.avg, step)
            test_writer.add_scalar("loss",loss.avg, step)
            test_writer.add_scalar("aux_loss",aux_loss.avg, step)
            for i,psnr in enumerate(psnr_list):
                test_writer.add_scalar(f"psnr_{i}",psnr.avg, step)
            last_index = len(psnr_list)-1
            test_writer.add_scalar("psnr",psnr_list[last_index].avg, step)
            test_writer.add_images('gen_recon', torch.cat((out_net["x_hat"][:4],d[:4]),dim=0), step) 
    
    print(
        f"iterations {iterations}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f} |"
        f"\tpsnr:",
        f",".join(f"{psnr.avg:.2f}" for psnr in psnr_list)
    )

    if logger.iteration>logger.max_iter:
        exit

    return loss.avg


def save_checkpoint(state, is_best, path):
    filename=os.path.join(path,"checkpoint.pth.tar")
    torch.save(state, filename)
    print("save to:%s\n"%filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path,"checkpoint_best_loss.pth.tar"))

def prepare_save(model="defualt",dataset="openimage",quality=1,ckpt=None,args=None):
    if ckpt is not None:
        root = os.path.split(os.path.split(ckpt)[0])[0]
        checkpoints_save = os.path.join(root,"checkpoints")
        figures_save = os.path.join(root, 'figures')
        tensorboard_runs = os.path.join(root, 'tensorboard')
        save_dirs = {"checkpoints_save": checkpoints_save,
        "figures_save":figures_save,
        "tensorboard_runs": tensorboard_runs
    }
    else:
        argv=sys.argv[1:]
        special_info = "{}_{}_q{}".format(model,dataset,quality)
        for opt in argv:
            if opt in "--scale":
                special_info = special_info+"_s%s"%(args.scale)
            if opt in "--loss":
                special_info = special_info+"_l%s"%(args.loss)
            if opt in "--lambda":
                special_info = special_info+"_%s"%(args.lmbda)
            # if opt in "--lambda":
            #     special_info = special_info+"_l%s"%(args.lambda)
                # special_info = special_info+"_lmd%f"%(args.lambda)
        save_dirs = setup_generic_signature(special_info)
    return save_dirs

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script")
    # yapf: disable
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        help='Training dataset path')
    parser.add_argument(
        '--dataname',
        type=str,
        default="openimage",
        help='Training dataset name')
    parser.add_argument(
        '--model',
        type=str,
        default = "AE",
        help='Training model')
    parser.add_argument(
        '-e',
        '--epochs',
        default=100,
        type=int,
        help='Number of epochs (default: %(default)s)')
    parser.add_argument(
        '-i',
        '--iterations',
        default=1e6,
        type=int,
        help='Number of iterations (default: %(default)s)')
    parser.add_argument(
        '-lr',
        '--learning-rate',
        default=1e-4,
        type=float,
        help='Learning rate (default: %(default)s)')
    parser.add_argument(
        '-n',
        '--num-workers',
        type=int,
        default=3,
        help='Dataloaders threads (default: %(default)s)')
    parser.add_argument(
        '--lambda',
        dest='lmbda',
        type=float,
        default=1e-2,
        help='Bit-rate distortion parameter (default: %(default)s)')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size (default: %(default)s)')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=4,
        help='Test batch size (default: %(default)s)')
    parser.add_argument(
        '--quality',
        type=int,
        default=1,
        help='quality (default: 1')
    parser.add_argument(
        '--aux-learning-rate',
        default=1e-3,
        help='Auxiliary loss learning rate (default: %(default)s)')
    parser.add_argument(
        '--patch-size',
        type=int,
        nargs=2,
        default=(256, 256),
        help='Size of the patches to be cropped (default: %(default)s)')
    parser.add_argument(
        '--cuda',
        action='store_true',
        help='Use cuda')
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save model to disk')
    parser.add_argument(
        '--seed',
        type=float,
        help='Set random seed for reproducibility')
    parser.add_argument(
        '--loss',
        type=str,
        default = "avg",
        help='loss type'
    )
    parser.add_argument(
        '--scale',
        type=int,
        default = 8,
        help='scale'
    )
    parser.add_argument(
        '--adaptive',
        type=str,
        default="spade",
        help='adaptive type'
    )
    parser.add_argument(
        '--resume',
        type=bool,
        default=False,
        help='Whether is resume')
    parser.add_argument(
        '--ckpt',
        type=str,
        help='ckpt path')
    
    
    parser.add_argument('--clip_max_norm',
                        default=0.1,
                        type=float,
                        help='gradient clipping max norm')
    
        
    
    # yapf: enable
    args = parser.parse_args(argv)
    return args

def load_checkpoint(path,device):
    ckpt = torch.load(path, map_location=device)
    return ckpt


def main(argv):
    cmd_args = parse_args(argv)

    if cmd_args.resume == True:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ckpt = load_checkpoint(cmd_args.ckpt,device)
        args = ckpt["args"]
        save_dirs = prepare_save(model=args.model,dataset=args.dataname,quality=args.quality,ckpt=cmd_args.ckpt)
    else:
        args = cmd_args
        save_dirs = prepare_save(model=args.model,dataset=args.dataname,quality=args.quality,args=args)

    logger = Logger(log_interval=100,test_inteval=100,save_dirs=save_dirs,max_iter=args.iterations)
    train_writer = SummaryWriter(os.path.join(save_dirs["tensorboard_runs"],"train"))
    test_writer = SummaryWriter(os.path.join(save_dirs["tensorboard_runs"],"test"))
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last= True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "AE":
        net = AutoEncoder()
    elif  args.model == "scale_bmshj2018_factorized":
        net = Scale_FactorizedPrior(N= 192, M=320,scale=args.scale)
    elif  args.model == "bmshj2018_hyperprior":
        net = bmshj2018_hyperprior(quality=args.quality)
    elif  args.model == "mbt2018_mean":
        net = mbt2018_mean(quality=args.quality)
    elif  args.model == "mbt2018":
        net = mbt2018(quality=args.quality)
    elif  args.model == "cheng2020_anchor":
        net = cheng2020_anchor(quality=args.quality)
    elif args.model == "MultiScale_FactorizedPrior":
        net = MultiScale_FactorizedPrior(N= 192, M=320,scale=args.scale)
    elif args.model == "MultiEB_Scale_FactorizedPrior":
        net = MultiEB_Scale_FactorizedPrior(N= 192, M=320,scale=args.scale)

    
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    aux_optimizer = optim.Adam(net.aux_parameters(), lr=args.aux_learning_rate)
    if args.loss == "avg":
        criterion = RateDistortionLoss(lmbda=args.lmbda)
    elif args.loss == "ID":
        criterion = InformationDistillationRateDistortionLoss(lmbda=args.lmbda)
    elif args.loss == "weight":
        criterion = WeightedRateDistortionLoss(lmbda=args.lmbda,scale=args.scale)

    best_loss = 1e10
    if cmd_args.resume == True:
        logger.iteration = ckpt["iteration"]
        best_loss = ckpt["loss"]
        net.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
        start_epoch = ckpt["epoch"]
    else:
        start_epoch = 0


    for epoch in range(start_epoch,args.epochs):
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            test_dataloader,
            args,
            writer=train_writer,
            test_writer=test_writer,
            logger=logger
        )

       


if __name__ == "__main__":
    main(sys.argv[1:])
