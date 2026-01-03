import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from utils.base import VOC_COLORMAP


_dataset_registry = {}


def register_dataset(name):
    """装饰器，用于注册数据集类"""

    def wrapper(cls):
        _dataset_registry[name] = cls
        return cls

    return wrapper


def load_dataset(name: str, batch_size, num_workers, **kwargs):
    if name not in _dataset_registry:
        raise ValueError(f"The 'name' must in {list(_dataset_registry.keys())}")

    dataset_class = _dataset_registry[name]
    train_iter = DataLoader(
        dataset_class(is_train=True, **kwargs), batch_size, shuffle=True, drop_last=True, num_workers=num_workers
    )
    test_iter = DataLoader(dataset_class(is_train=False, **kwargs), batch_size, drop_last=True, num_workers=num_workers)
    return train_iter, test_iter


@register_dataset("VOC2012")
class VOC2012(Dataset):
    def __init__(self, crop_size, is_train: bool = True):
        super().__init__()
        self.crop_size = crop_size
        self.transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        features, labels = self.read_voc_images("/root/autodl-tmp/d2l/dataset/VOCdevkit/VOC2012", is_train=is_train)
        self.features = [self.normalize_image(feature) for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap = self.voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def __getitem__(self, idx):
        feature, label = self.voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        return feature, self.voc_label_indices(label, self.colormap)

    def __len__(self):
        return len(self.features)

    def read_voc_images(self, voc_dir, is_train=True):
        txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt' if is_train else 'val.txt')
        mode = torchvision.io.image.ImageReadMode.RGB
        with open(txt_fname, 'r') as f:
            images = f.read().split()
        features, labels = [], []
        for i, fname in enumerate(images):
            features.append(torchvision.io.read_image(os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')))
            labels.append(torchvision.io.read_image(os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'), mode))
        return features, labels

    def filter(self, imgs):
        _imgs = []
        for img in imgs:
            if img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1]:
                _imgs.append(img)
        return _imgs

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def voc_colormap2label(self):
        """
        构建 color map
        ARGS:
            - None
        RETURN:
            1. 长度为256**3的向量, 其中RGB经过哈希计算后结果的对应位置存放的时classes的索引
        """
        colormap2label = torch.zeros(256**3, dtype=torch.long)
        for i, colormap in enumerate(VOC_COLORMAP):
            colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
        return colormap2label

    def voc_label_indices(self, img, colormap):
        """
        图像每个像素位置RGB值通过 color map 映射到类别的索引
        ARGS:
            - img: 图像(H, W, C)格式的三维张量
            - colormap: 存放RGB哈希结果于其对应类别索引的映射表
        RETURN:
            1. 每个像素对应的类别索引
        """
        img = img.permute(1, 2, 0).numpy().astype('int32')
        idx = (img[:, :, 0] * 256 + img[:, :, 1]) * 256 + img[:, :, 2]
        return colormap[idx]

    def voc_rand_crop(self, feature, label, height, width):
        """
        随机裁剪特征和标签图像
        """
        rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
        feature = torchvision.transforms.functional.crop(feature, *rect)
        label = torchvision.transforms.functional.crop(label, *rect)
        return feature, label
