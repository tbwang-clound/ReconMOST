"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util_v2 import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    ) # 默认没有用checkponit——细化到网络结构里发现根本没有使用
    
    # 从这里load_state_dict(params)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()  # 设置为evaluation模式(not update)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        # 从这里开始采样
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )  # sample_fn表示采样函数——区别是ddim_sample_loop多了一个ddim_loss（我还没细看）
        sample = sample_fn(
            model,
            #(args.batch_size, 3, args.image_size, args.image_size),
            (args.batch_size, args.in_channels, args.image_size_H, args.image_size_W),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs, 
        )  # 模型参数，需要更改channel
        # inverse数据预处理
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # sample = ((sample + 1) * 22.5 - 5).clamp(-5, 40).to(th.uint8)  # unit8就是0-255，-5会变成255
        sample = ((sample + 1) * 22.5).clamp(0, 45).to(th.uint8)  # 截断到0-45，-5在显示 的时候再处理.不能在这里-5，因为减了5还是unit8负数->255
        sample = sample.permute(0, 2, 3, 1)  # batch不变，将channel(通道数)放到最后，可以完全不看最后那个嘞
        sample = sample.contiguous() # 保证张量在内存中连续

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        out_dir = os.path.dirname(os.environ.get("DIFFUSION_SAMPLE_LOGDIR", logger.get_dir()))
        os.makedirs(out_dir, exist_ok=True)  # 确保目录存在
        out_path = os.path.join(out_dir, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)  # 保存为npz格式？

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
        model_path='',  # 待补充model_path
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
