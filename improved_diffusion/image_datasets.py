from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, ocean=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    
    if ocean == True:
        all_files = _list_npy_files_recursively()
    else:
        if not data_dir:
            raise ValueError("unspecified data directory")
        all_files = _list_image_files_recursively(data_dir) # 返回路径名
    
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

# 更新，新增加npy类型，加入文件路径
def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif","npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def _list_npy_files_recursively():
    # 只是为了保存下每一个图片的路径
    data_dir = '/home/bingxing2/ailab/scxlab0052/yysong/improved-diffusion-main/TOSdataset'
    # model_name = ['FGOALS-f3-L', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'FIO-ESM-2-0']
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1] # 获得扩展名
        if "." in entry and ext.lower() in ["npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution   # image_size 180*360
        self.local_images = image_paths[shard:][::num_shards] # 后者是贴片的意思（步长）
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]

        # with bf.BlobFile(path, "rb") as f:
        #     pil_image = Image.open(f)
        #     pil_image.load()

        # # We are not on a new enough PIL to support the `reducing_gap`
        # # argument, which uses BOX downsampling at powers of two first.
        # # Thus, we do it by hand to improve downsample quality.
        # while min(*pil_image.size) >= 2 * self.resolution:
        #     pil_image = pil_image.resize(
        #         tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        #     )

        # scale = self.resolution / min(*pil_image.size)
        # pil_image = pil_image.resize(
        #     tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        # )

        # arr = np.array(pil_image.convert("RGB"))  
        ext = path.split(".")[-1].lower()

        if ext == "npy":
            # 调整变成和图像一样的格式
            with bf.BlobFile(path, "rb") as f:
                arr = np.load(f) # 返回结果为array
                # 将每个值重复3次放在最后一个维度上
                # arr = np.expand_dims(arr, axis=2)
                # arr = np.repeat(arr, 3, axis=2)
                # 数值范围在-5~40: 0.04x-0.8
                #arr = arr.astype(np.float32) * 0.04 - 0.8
                arr = np.nan_to_num(arr, nan=0.0)
                pil_image = Image.fromarray((arr + 5).astype(np.uint8))
        else:
            with bf.BlobFile(path, "rb") as f:
                pil_image = Image.open(f)
                pil_image.load()

        while min(*pil_image.size) >= 2 * self.resolution:
            # 检查图像的最小尺寸是否大于分辨率（输入大小）的两倍。是则不断缩小一般直到最小的尺寸小于分辨率的两倍
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        # 再次缩放图像，使得最小的尺寸等于分辨率
        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        # 转换成RGB格式再转换成array,HWC
        arr = np.array(pil_image.convert("RGB"))
        arr = arr.astype(np.float32) / 127.5 - 1  # 结果在-1 ~ 1，归一化的结果

        # 裁剪成只剩中心部分了
        crop_y = (arr.shape[0] - self.resolution) // 2  
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        # 最终的结果一定是先与原图像等比例大小，用插值的方法重采样来降低分辨率，然后再crop到resolution*resolution的大小
        # 使用高分辨率时还要小心，小的那条边不要超出范围

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        # 返回第一维是RGB，然后是HW
        return np.transpose(arr, [2, 0, 1]), out_dict
