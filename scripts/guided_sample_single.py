"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

# 使用带噪声的图像分类器的梯度指导采样过程，以便生成更逼真的图像

import argparse
import os
import time

import numpy as np
import torch as th
import blobfile as bf
import torch.distributed as dist
import torch.nn.functional as F

import xarray as xr

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util_v2 import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    # classifier_defaults,  # Added
    create_model_and_diffusion,
    # create_classifier, # Added
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.guided_util import *

def get_gaussian_kernel(size, sigma):
    # 生成3D高斯核，可考虑在外面只生成一次，不然浪费计算时间
    coords = th.arange(size, dtype=th.float32)
    coords -= size // 2
    g = th.exp(-(coords**2) / (2 * sigma**2))  # 一维高斯核
    g = g.outer(g)  # 外积生成二维高斯核
    # return (g / g.sum()).view(1, 1, size, size)  # 归一化
    return g.view(1, 1, size, size)

def get_gaussian_kernel_3d(size, sigma):
    # 生成3D高斯核
    coords = th.arange(size, dtype=th.float32)
    coords -= size // 2
    g = th.exp(-(coords**2) / (2 * sigma**2))  # 一维高斯核
    g_3d = g[:, None, None] * g[None, :, None] * g[None, None, :]  # 外积生成三维高斯核
    return  g_3d.view(1, 1, size, size, size)  # 返回形状为 (1, 1, D, H, W) 的核


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())  # Unet Model 和 Gaussian Diffusion
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()  # load unet模型参数并设置为评估模式，不会更新参数

    logger.log("Loading sparse data for guiding...")
    # 需要用新的读取方式 ds那个，可以直接写进来
    # guided_arr_dict = get_guided_arr_dict(args.sparse_data_path, args.in_channels) 
    guided_arr_dict = {}
    for file in bf.listdir(args.sparse_data_path):
        # file only contains the name of the file, not the full path
        if file.endswith(".nc"):
            path = os.path.join(args.sparse_data_path,file)
            ds = xr.open_dataset(path)
            arr = ds.thetao.values  # 42层，173*360  -83-89
            arr = arr.astype(np.float32)  # 保留nan [-5,40]
        if len(arr.shape) == 4:
            # reshape成CHW，把第一维去掉
            arr = arr.reshape(arr.shape[1], arr.shape[2], arr.shape[3])
        guided_arr_dict[str(file)[0:-3]] = arr  # .npy->.nc
    # print(guided_arr_dict.keys())  # 应该是很多个
    logger.log("Successfully load the guided data!")

    # 只在平面区域soft，配合cond_fn使用
    softmask_kernel = get_gaussian_kernel(5, 1.0).float().to("cuda") 

    # 在3D区域soft，配合cond_fn_3d使用
    # softmask_kernel = get_gaussian_kernel_3d(5, 0.5).float().to("cuda")  # 3D高斯核
    print(softmask_kernel)

    # 定义两个需要使用的函数
    def cond_fn(x, t, p_mean_var, y=None):
        # 输入的x大小应该为 BCHW tensor
        # loss函数为：MSE(y,x0) + Q(x0) maybe 
        assert y is not None
        x = p_mean_var['pred_xstart'] # x0
        s = args.grad_scale
        gradient = 2 * ( y - x )   # y中为nan的地方也是nan
        gradient = th.nan_to_num(gradient, nan=0.0)  # 将nan转换为0
        # 返回也是tensor形式
        if args.use_softmask:
            size = softmask_kernel.shape[-1]
            gradient = F.conv2d(
            gradient, 
            softmask_kernel.expand(gradient.shape[1], 1, size, size), # 卷积核的形状
            stride=1,
            padding= size // 2,
            groups=gradient.shape[1] # 每个通道C独立卷积——使用深度卷积（分成通道个组来卷积）
            )
        return gradient * s
        # 另外添加softmax——类似对于每一个x0卷积一个高斯核

    
    def cond_fn_3d(x, t, p_mean_var, y=None):
        assert y is not None
        x = p_mean_var['pred_xstart']  # x0
        s = args.grad_scale
        gradient = 2 * (y - x)  # 计算梯度
        gradient = th.nan_to_num(gradient, nan=0.0)  # 将nan转换为0

        if args.use_softmask:
            size = softmask_kernel.shape[-1]
            gradient = gradient.unsqueeze(1)  # 添加一个伪深度维度，形状变为 (B, 1, C, H, W)
            gradient = F.conv3d(
                gradient, 
                softmask_kernel,  # 卷积核的形状为 (1, 1, size, size, size)
                stride=1,
                padding=size // 2,
                groups=1  # 跨通道卷积
            )  # B1CHW 
            gradient = gradient.squeeze(1)  # 去掉伪深度维度，恢复为 (B, C, H, W)
        return gradient * s    
    

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)
        # 但是class_cond默认为False时y是None

    # outputdir
    date = time.strftime("%m%d")
    if args.dynamic_guided:
        config = f"dyn_next={args.dynamic_guided_with_next}_r={args.guided_rate}_sigma={args.use_sigma}"
    else:
        config = f"s={args.grad_scale}_r={args.guided_rate}_loss={args.loss_model}_softmask={args.use_softmask}_sigma={args.use_sigma}"

    out_dir = os.path.dirname(os.environ.get("DIFFUSION_SAMPLE_LOGDIR", logger.get_dir()))
    out_dir=os.path.join(out_dir, date, config)
    os.makedirs(out_dir, exist_ok=True)


    logger.log("sampling...")
    
    loss_preds = []
    loss_guideds = []
    losses = []
    # 开始采样
    for key, guided_arr in guided_arr_dict.items():
        all_images = []
        all_loss_pred = []
        all_loss_guided = []
        all_loss = []
        logger.log(f"sampling {key}...")
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}  # 模型参数字典
            # 按照guided_rate拆分成指导和评估两部分,并扩展成batch_size个
            guided_y, eval_y = split_guided_eval_batch_size(args.batch_size, guided_arr, args.guided_rate) # -5,40
            model_kwargs["y"] = normalization(guided_y)  # 归一化到-1,1
            # 传入指导观测图像
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )  # 定义采样函数，是否使用DDIM
            sample = sample_fn(
                model_fn,  # 模型：改动1！！！
                (args.batch_size, args.in_channels, args.image_size_H, args.image_size_W),  # 形状，第二个已修改
                clip_denoised=args.clip_denoised,  # 是否裁剪去噪后的图片，默认为True 
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,  # 条件：改动2！！！
                use_sigma = args.use_sigma,  # sigma?=0
                dynamic_guided=args.dynamic_guided,  # 是否动态调整sigma
                dynamic_guided_with_next=args.dynamic_guided_with_next,  # 是否使用下一步的sample
                device=dist_util.dev(),
            )  # 返回采样结果


            sample = ((sample + 1) * 22.5 - 5 ).clamp(-5, 40)  # 截断到-5-40，浮点数
            sample = sample.permute(0, 2, 3, 1)  # BHWC  图片存储是HWC
            sample = sample.contiguous() 

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL

            # 计算一个batch里面所有图片的loss，按通道返回
            loss_pred = calculate_loss(sample, eval_y, args.loss_model)
            loss_guided = calculate_loss(sample, guided_y, args.loss_model)
            loss_total = (1-args.guided_rate)*loss_pred + args.guided_rate * loss_guided
            all_loss_pred.append(loss_pred)
            all_loss_guided.append(loss_guided)
            all_loss.append(loss_total)

            # 将sample图像转移到cpu上,并且将此次生成的batch_size张图片加入all_images中作为一个batch
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

            logger.log(f"created {len(all_images) * args.batch_size} samples")

        # 一张图片的结果总结
        arr = np.concatenate(all_images, axis=0)
        # arr = arr[: args.num_samples] # 全保留，不截取了，不然不好办，后面加一个采样数和batch是否是倍数关系的检测
        # 计算该观测所有采样结果的loss
        pred_loss = np.mean(all_loss_pred, axis=0)
        guided_loss = np.mean(all_loss_guided, axis=0)
        total_loss = np.mean(all_loss, axis=0)
        loss_preds.append(pred_loss)
        loss_guideds.append(guided_loss)
        losses.append(total_loss)
        
        # 一张图片的结果打印
        if dist.get_rank() == 0:
            logger.log(f"Loss of {key} pred_loss: {pred_loss}, guided_loss: {guided_loss}, total_loss: {total_loss}")
            shape_str = "x".join([str(x) for x in arr.shape]) 
            # outputpath
            out_path = os.path.join(out_dir, f"{key}_sample{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, arr)
            logger.log(f"sampling {key} complete")

    # 所有图片的结果
    dist.barrier() # 同步所有进程(等待所有进程完成)
    logger.log("Complete All Sample!")
    logger.log(f"Compute Total Loss: pred_loss: {np.mean(loss_preds, axis=0)}, guided_loss: {np.mean(loss_guideds, axis=0)}, total_loss: {np.mean(losses, axis=0)}")


def create_argparser():
    defaults = dict(
        image_size_H = 173,
        image_size_W = 360,
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        use_sigma=True,
        model_path="",
        sparse_data_path="",
        grad_scale=1.0,  # when 0: sample from the base diffusion model
        use_softmask=False,
        dynamic_guided = True,
        dynamic_guided_with_next = False,
        guided_rate=0.6,
        loss_model="l1",
        use_fp16=False,
    )
    defaults.update(model_and_diffusion_defaults())
    # defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()

    # if dist.is_initialized():
    #     dist.destroy_process_group()
