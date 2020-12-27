from torchsummary import summary
import math
import io
import torch
from torchvision import transforms
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from pytorch_msssim import ms_ssim
from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
metric = 'mse'  # only pre-trained model for mse are available for now


from examples.train import *

test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )
test_dataset = ImageFolder("../data", split="test", transform=test_transforms)
test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
    )

def test2(iterations, test_dataloader, model, criterion,test_writer=None,logger=None):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model.ae(d)
#             x = d
#             y = model.g_a(x)
#             z = model.h_a(torch.abs(y))
#             z_hat, z_likelihoods = model.entropy_bottleneck(z)
#             scales_hat = model.h_s(z_hat)
#             y_hat, y_likelihoods = model.gaussian_conditional(y, scales_hat)
#             x_hat = model.g_s(y)

#             out_net={
#                 "x_hat": x_hat,
#                 "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
#             }

            out_criterion = criterion(out_net, d)
            out_net["x_hat"] = torch.clamp(out_net["x_hat"],min=0.,max=1.)
            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(compute_psnr(out_net["x_hat"]*255.0,d*255.0))

        if test_writer is not None:
            step = logger.iteration
            test_writer.add_scalar("mse_loss",mse_loss.avg, step)
            test_writer.add_scalar("bpp_loss",bpp_loss.avg, step)
            test_writer.add_scalar("loss",loss.avg, step)
            test_writer.add_scalar("aux_loss",aux_loss.avg, step),
            test_writer.add_scalar("psnr",psnr.avg, step),
            test_writer.add_images('gen_recon', torch.cat((out_net["x_hat"][:4],d[:4]),dim=0), step) 
    
    print(
        f"\niterations {iterations}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f} |"
        f"\tpsnr: {psnr.avg:.3f} \n"
    )


    return bpp_loss.avg,psnr.avg

def test(iterations, test_dataloader, model, criterion,test_writer=None,logger=None):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()

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
            psnr.update(compute_psnr(out_net["x_hat"]*255.0,d*255.0))

        if test_writer is not None:
            step = logger.iteration
            test_writer.add_scalar("mse_loss",mse_loss.avg, step)
            test_writer.add_scalar("bpp_loss",bpp_loss.avg, step)
            test_writer.add_scalar("loss",loss.avg, step)
            test_writer.add_scalar("aux_loss",aux_loss.avg, step),
            test_writer.add_scalar("psnr",psnr.avg, step),
            test_writer.add_images('gen_recon', torch.cat((out_net["x_hat"][:4],d[:4]),dim=0), step) 
    
    print(
        f"\niterations {iterations}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f} |"
        f"\tpsnr: {psnr.avg:.3f} \n"
    )

    return bpp_loss.avg,psnr.avg

networks = {
    'bmshj2018-factorized': bmshj2018_factorized,
    'bmshj2018-hyperprior': bmshj2018_hyperprior,
    'mbt2018-mean': mbt2018_mean,
    'mbt2018': mbt2018,
    'cheng2020-anchor': cheng2020_anchor,
}
for arch,num in zip(["bmshj2018-factorized","bmshj2018-hyperprior","mbt2018-mean","mbt2018","cheng2020-anchor"],[8,8,8,8,6]):
    model_list = []
    for i in range(num):
        model_list.append(networks[arch](quality=i+1, pretrained=True).eval().to(device))

    bpp_list = []
    bpp_list2 = []
    psnr_list = []
    psnr_list2 = []
    for model in model_list:
        bpp,psnr = test(iterations=0, test_dataloader=test_dataloader, model=model, criterion=RateDistortionLoss(lmbda=1e-3))
        bpp_list.append(bpp)
        psnr_list.append(psnr)
        bpp2,psnr2 = test2(iterations=0, test_dataloader=test_dataloader, model=model, criterion=RateDistortionLoss(lmbda=1e-3))
        bpp_list2.append(bpp2)
        psnr_list2.append(psnr2)

    plt.figure(figsize=(10,10))
    plt.plot(bpp_list,psnr_list,marker='v',label="w entropy_bottleneck")
    plt.plot(bpp_list2,psnr_list2,marker='v',label="w/o entropy_bottleneck")
    plt.xlabel('bpp')
    plt.ylabel('PSNR')
    plt.title("%s_kodak"%arch)
    plt.legend()
    plt.show()
