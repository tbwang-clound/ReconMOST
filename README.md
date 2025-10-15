# ReconMOST
adjusted_diffusion
## Usage

This section of the README walks through how to train and sample from a model.

### Installation

Clone this repository and navigate to it in your terminal. Then run:

```
pip install -e .
```

This should install the `improved_diffusion` python package that the scripts depend on.

### Preparing Data

The training code reads images from a directory of image files. In the [datasets](datasets) folder, we have provided instructions/scripts for preparing these directories for ImageNet, LSUN bedrooms, and CIFAR-10.

For creating your own dataset, simply dump all of your images into a directory with ".jpg", ".jpeg", or ".png" extensions. If you wish to train a class-conditional model, name the files like "mylabel1_XXX.jpg", "mylabel2_YYY.jpg", etc., so that the data loader knows that "mylabel1" and "mylabel2" are the labels. Subdirectories will automatically be enumerated as well, so the images can be organized into a recursive structure (although the directory names will be ignored, and the underscore prefixes are used as names).

The images will automatically be scaled and center-cropped by the data-loading pipeline. Simply pass `--data_dir path/to/images` to the training script, and it will take care of the rest.

### Training

To train your model, you should first decide some hyperparameters. We will split up our hyperparameters into three groups: model architecture, diffusion process, and training flags. Here is a training example:

```
DATA_DIR=YOUR_DATA_PATH
MODEL_FLAGS="--in_channels 42 --image_size 180 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 4"
export DIFFUSION_BLOB_LOGDIR=YOUR_SAVE_PATH
python ./scripts/image_train_v2.py --data_dir $DATA_DIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

## Sampling

The above training script saves checkpoints to `.pt` files in the logging directory. These checkpoints will have names like `ema_0.9999_200000.pt` and `model200000.pt`. You will likely want to sample from the EMA models, since those produce much better samples.

Once you have a path to your model, you can generate a large batch of samples like so:

```
MODEL_PATH=YOUR_MODEL_PATH
SPARSE_DATA_PATH=YOUR_DATA_PATH
ADJUST_FLAGS="--num_samples 4 --use_sigma False --grad_scale 4 --use_softmask False --guided_rate 0.075 --loss_model l2"
export DIFFUSION_SAMPLE_LOGDIR=YOUR_SAVE_PATH
MODEL_FLAGS="--in_channels 42 --image_size 180 --image_size_H 173 --image_size_W 360 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --use_ddim False  --clip_denoised True"
SAMPLE_FLAGS="--batch_size 4"
python ./scripts/guided_sample_analysis.py --model_path $MODEL_PATH --sparse_data_path $SPARSE_DATA_PATH $MODEL_FLAGS $DIFFUSION_FLAGS  $SAMPLE_FLAGS $ADJUST_FLAGS
```

```bash
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"
```

### ReconMOST数据准备
此处以使用FGOALS-f3-l作为训练数据，EN4作为测试数据为例

下载下来的FGOALS-f3-l文件名如thetao_Omon_FGOALS-f3-L_historical_r1i1p1f1_gn_185001-189912.nc，每个.nc文件是很多年的数据，比如这个文件是50年的，那么尺寸是（600（600个月），30（30个深度），218（纬度），360（经度）），需要把这些文件放在文件夹A

下载的EN4数据集解压后是按月份分成不同的.nc文件的，把他们放在文件夹B

然后对于A,B，配置输出路径后使用脚本prepare_data.py处理数据（统一变量名，对不同海温深度线性插值到42层），分别获得处理后的新文件夹C,D

接下来需要手动修改一下文件夹C的格式，根据代码要求，文件夹C（存放处理完后的数据）应该是这样的格式，所以需要在文件夹C下面创建一个新的文件夹，可以叫FGOALS-f3-L，然后把所属的月份数据全部放进去，然后在文件夹C里创建一个 zz_validation_holdout，因为按字母排序的最后一个文件夹，将被自动忽略
```
/path/to/your/processed_training_data/  <-- 训练时这个路径传给 --data_dir
│
├── BCC-CSM2-MR/                  # 第一个数据源（例如一个CMIP6模型）
│   ├── BCC-CSM2-MR_..._185001.nc
│   └── BCC-CSM2-MR_..._185002.nc
│   └── ...
│
├── CanESM5/                      # 第二个数据源
│   ├── CanESM5_..._185001.nc
│   └── CanESM5_..._185002.nc
│   └── ...
│
├── FGOALS-f3-L/                  # 第三个数据源
│   ├── FGOALS-f3-L_..._185001.nc
│   └── FGOALS-f3-L_..._185002.nc
│   └── ...
│
├── ... (其他模型的文件夹) ...
│
└── zz_validation_holdout/      # 按字母排序的最后一个文件夹，将被自动忽略
    └── (可以是空的，或存放验证集文件)
```

训练命令参考：

```
# 设置日志和模型检查点的保存路径
export DIFFUSION_BLOB_LOGDIR="/data/coding/data/ReconMOST_training_logs"

# 启动训练
python ./scripts/image_train_v2.py \
    --data_dir /data/coding/data/ReconMOST_train_processed_en4grid \
    --in_channels 42 \
    --image_size 180 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --attention_resolutions "16,8" \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --lr 1e-4 \
    --batch_size 4 \
    --save_interval 10000 \
    --log_interval 100 \
    --schedule_sampler uniform
```

推理命令参考：
```
# 设置本次推理的日志和结果保存路径
export OPENAI_LOGDIR="/data/coding/data/ReconMOST_sampling_results_10000"

# 启动推理
python ./scripts/guided_sample_analysis.py \
    --model_path /data/coding/data/ReconMOST_training_logs/model010000.pt \
    --sparse_data_path /data/coding/data/ReconMOST_testdata_processed_en4grid \
    --in_channels 42 \
    --image_size 180 \
    --num_channels 128 \
    --num_res_blocks 3 \
    --attention_resolutions "16,8" \
    --grad_scale 5.0 \
    --guided_rate 0.075 \
    --use_sigma False \
    --use_softmask True \
    --num_samples 16 \
    --batch_size 4 \
    --loss_model l1
```

为了方便数据下载，可以从以下百度网盘链接中获取所需的数据：

https://pan.baidu.com/s/1RM4LH7KeY0T84Ux8gKXPzw?pwd=6666 
