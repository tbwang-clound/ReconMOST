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
from improved_diffusion.guided_util import get_guided_arr_dict


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

    # logger.log("loading classifier...")
    # #classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    # classifier.load_state_dict(
    #     dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    # )
    # classifier.to(dist_util.dev())
    # if args.classifier_use_fp16:
    #     classifier.convert_to_fp16()
    # classifier.eval()  # load分类器参数，并且分类器处于评估模式，即不更新参数
    logger.log("Loading sparse data for guiding...")
    path = args.sparse_data_path
    guided_arr_dict = {}
    for file in bf.listdir(path):
        # file only contains the name of the file, not the full path
        if file.endswith(".npy"):
            with bf.BlobFile(os.path.join(path,file), "rb") as f:
                arr = np.load(f) # 返回结果为 array [180, 360, 38] HWC
                # arr = np.nan_to_num(arr, nan=0.0) # 不能转换，不然无法判断哪里没有值
                arr = 2 * (arr + 5) / 45 - 1   # rescale [-1, 1]
                arr = arr.astype(np.float32)
                arr = np.transpose(arr, (2, 0, 1))[0]  # 海表数据CHW
                arr = np.expand_dims(arr, axis=0)
                guided_arr_dict[str(file)[0:-4]] = arr
    
    guided_arr_dict = get_guided_arr_dict(args.sparse_data_path)  # 获取指导数组字典
    grad_scale = args.guided_scale  # 梯度缩放因子

    # 定义两个需要使用的函数
    def cond_fn(x, t, s=1, y=None):
        # 输入的x大小应该为 BCHW tensor
        # loss函数为：MSE(y,x0) + Q(x0) maybe 
        assert y is not None
        # mask = ~np.isnan(y)  # mask为y中非nan的部分
        # gradient = 2 * ( x - y )  # 梯度为2*(x-y)，y中为nan的地方也是nan
        gradient = 2 * ( y - x ) 
        gradient = th.nan_to_num(gradient, nan=0.0)  # 将nan转换为0
        # 返回也是tensor形式
        return gradient * s

    def cond_fn_l1(x, t, y=None):
        # loss函数为：ABS(y,x0) + Q(x0) maybe 
        assert y is not None
        gradient = 2 * ( x - y )  # 梯度为2*(x-y)再加上一个mask
        # p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return gradient

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)
        # 但是class_cond默认为False时y是None


    logger.log("sampling...")
    all_images = []
    # 开始采样
    for key, guided_arr in guided_arr_dict.items():
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}  # 模型参数字典
            # NUM_CLASSES = len(guided_arr_dict.keys)  # 类别数
            # classes = th.randint(
            #     low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            # )  # 随机生成一个batch_size大小的类别标签
            # model_kwargs["y"] = classes  # 类别标签
            tensor_y = th.from_numpy(np.stack([guided_arr]*args.batch_size, axis=0)).float().to("cuda")
            model_kwargs["s"] = grad_scale  # 类别标签的缩放因子
            model_kwargs["y"] = tensor_y
            # 传入类别标签，改为传入类别图像
            # 放上图片本身，大小为(batch_size, in_channel, args.image_size, args.image_size)

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
                device=dist_util.dev(),
            )  # 返回采样结果


            sample = ((sample + 1) * 22.5 - 5 ).clamp(-5, 40)  # 截断到-5-40，浮点数
            sample = sample.permute(0, 2, 3, 1)  # HWC  
            sample = sample.contiguous() 

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

            ######## 处理类别，但应该不是类别(不处理直接存，一次先只测试一张图)
            # gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())] # 搞一个形状和classes一样的全0张量，一共有dist.get_world_size()个
            # dist.all_gather(gathered_labels, classes) # 将classes收集到gathered_labels中（复制？）
            # all_labels.extend([labels.cpu().numpy() for labels in gathered_labels]) # 将gathered_labels转换为numpy数组，然后加入all_labels
            ##########

            logger.log(f"created {len(all_images) * args.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        ##########
        # label_arr = np.concatenate(all_labels, axis=0) # 将all_labels拼接成一个数组
        # label_arr = label_arr[: args.num_samples]  # 取前args.num_samples个
        ##########
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_dir = os.path.dirname(os.environ.get("DIFFUSION_SAMPLE_LOGDIR", logger.get_dir()))
            os.makedirs(out_dir, exist_ok=True) 
            date = time.strftime("%Y%m%d-%H%M")
            out_path = os.path.join(out_dir, f"{date}_{key}_sample{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, arr)
            logger.log(f"sampling {key} complete")

        #  np.savez(out_path, arr, label_arr)  # label_arr?
        # 按照label_arr的label存储图片，第一维为label_idx，第二维为image_idx，第三维为image (pickle)，需要改储存方式
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        image_size_H = 180,
        image_size_W = 360,
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        use_sigma=True,
        model_path="",
        sparse_data_path="",
        guided_scale=1.0,  # when 0: sample from the base diffusion model
        use_fp16=False,
    )
    defaults.update(model_and_diffusion_defaults())
    # defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
