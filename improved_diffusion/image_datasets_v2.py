# from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import xarray as xr

# 不是很需要参考Guided Diffusion的结构

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
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
    if not data_dir:
        raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)
    all_files, train_entries = _list_multi_mode_train_recursively(data_dir)
    # print("Load Data from Mode: ", data_dir)
    # print("Choose Train Entries: ", train_entries)
    for mode in train_entries:
        print("Load Data from Mode: ", mode)
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


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        # entry是文件(夹)名
        full_path = bf.join(data_dir, entry)  # path更新为输入目录/entry
        ext = entry.split(".")[-1]
        # add NPY file.
        # if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy", "nc"]:
        if "." in entry and ext.lower() in ['nc']:
            results.append(full_path)
        elif bf.isdir(full_path):
            # 如果又是一个文件夹，递归调用
            results.extend(_list_image_files_recursively(full_path))
    return results

# use in single mode train(split every mode into train and test)
def _list_files_split_train_recursively(data_dir):
    # data_dir结尾为模式名，内涵不同初始化，留下最后一个文件夹
    results = []
    train_entries = []
    for entry in sorted(bf.listdir(data_dir))[0:-1]:
        # entry是模式名，留下最后一个作为测试集，按文件名的字典序排序
        train_entries.append(entry)
        full_path = bf.join(data_dir, entry)
        results.extend(_list_image_files_recursively(full_path))
    return results, train_entries


# use in multi mode train(ablation 2)
def _list_multi_mode_train_recursively(data_dir):
    # data_dir结尾为模式名，内涵不同初始化，留下最后一个文件夹
    results = []
    train_entries = []
    modes = ['FIO-ESM-2-0','BCC-CSM2-MR','MRI-ESM2-0','CanESM5','IPSL-CM6A-LR','FGOALS-g3','FGOALS-f3-L']
    for mode in modes:
        # entry是模式名，留下最后一个作为测试集，按文件名的字典序排序
        full_path = bf.join(data_dir, mode)
        train_entries.append(full_path)
        results.extend(_list_image_files_recursively(full_path))
    return results, train_entries


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution  # image_size 180*360
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]

        # 单层.npy
        if path.endswith(".npy"):
            with bf.BlobFile(path, "rb") as f:
                # pil_image = Image.open(f)
                # pil_image.load()
                arr = np.load(f) # 返回结果为 array [180, 360]
                arr = np.nan_to_num(arr, nan=0.0)
                arr = 2 * (arr + 5) / 45 - 1   # rescale [-1, 1]
                arr = arr.astype(np.float32)

        # 多层.nc
        elif path.endswith(".nc"):
            ds = xr.open_dataset(path)
            arr = ds.thetao.values  # 42层，173*360  -83-89
            arr = np.nan_to_num(arr, nan=0.0)
            arr = 2 * (arr + 5) / 45 - 1   # [-5, 40] rescale to [-1, 1]
            arr = arr.astype(np.float32)
            
        out_dict = {}
        # if self.local_classes is not None:
        #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        # 增加一个维度。没有crop。
        if len(arr.shape) == 2:
            # reshape成CHW
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])
        if len(arr.shape) == 4:
            # reshape成CHW，把第一维去掉
            arr = arr.reshape(arr.shape[1], arr.shape[2], arr.shape[3])
        return arr, out_dict
