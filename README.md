# CLD: Unsupervised Feature Learning by Cross-Level Instance-Group Discrimination.

## Requirement
 
### Packages
* Python >= 3.7, < 3.9
* PyTorch >= 1.6, <= 1.10
* pandas
* numpy
* [apex](https://github.com/NVIDIA/apex) (optional, unless using mixed precision training)

```shell
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## 数据准备

```shell
python generate_images.py --image_dir $IMG_DIR --stride 0.5 --xml_dir $XML_DIR --save_dir data/train/images
```
- `image_dir`: 表示遥感图像所在是文件夹，将对此文件夹下所有的图片进行裁剪，生成训练图像
- `xml_dir` : 已经标注好的图像的xml文件存储位置。是为了统计标注目标框的大小，从而计算得到crop 图像的大小。代码中是计算标注框的高、宽的平均值，然后乘以1.2进行crop
- `stride`: 裁剪图像的步长。裁剪过程中，得到裁剪图像大小后，乘以stride得到在原图中裁剪的高、宽的步长。默认0.5
- `save_dir`: 裁剪图像的存放位置。不要修改，与训练代码已经绑定。

## 开始训练

```shll
# 默认是8卡训练，如需修改卡数，则直接变train.sh
sh train.sh
```

