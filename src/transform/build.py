import torch.nn as nn
from torchvision import transforms 
import torchvision.transforms.functional as F

from .augment import build_augmentation


image_process = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

mask_process = transforms.PILToTensor()


class SegTransform: 
    def __init__(self, img_size=None, aug=None):
        if img_size is not None:
            self.resize_image = transforms.Resize(img_size, interpolation=F.InterpolationMode.BILINEAR) 
            self.resize_mask = transforms.Resize(img_size, interpolation=F.InterpolationMode.NEAREST) 
        else:
            self.resize_image = nn.Identity()
            self.resize_mask = nn.Identity()
        self.aug = aug

    def __call__(self, image, mask, label=None, bboxes=None):
        # data aug
        if self.aug is not None:
            image, mask = self.aug(image, mask)

        # resize
        image = self.resize_image(image)
        if mask is not None:
            mask = self.resize_mask(mask)

        # image preprocess
        image = image_process(image)

        # mask preprocess
        if mask is not None:
            mask = mask_process(mask).squeeze(0)
        return image, mask



class ClsTransform:
    def __init__(self, img_size=None, aug=None):
        if img_size is not None:
            self.resize_image = transforms.Resize(img_size, interpolation=F.InterpolationMode.BILINEAR)
        else: 
            self.resize_image = nn.Identity()
        self.aug = aug

    def __call__(self, image, mask, label, bboxes):
        # data aug
        if self.aug is not None:
            image, label = self.aug(image, mask, label, bboxes)

        image = self.resize_image(image)
        image = image_process(image)
        return image, label


def build_seg_transform(is_train, **kwargs):
    img_size = kwargs.pop("img_size")
    augment_kwargs_list = kwargs.pop("aug", None)
    if augment_kwargs_list is None:
        aug = None
    else:
        aug = build_augmentation(augment_kwargs_list) if is_train else None
    return SegTransform(img_size=img_size, aug=aug)


def build_cls_transform(is_train, **kwargs):
    img_size = kwargs.pop("img_size")
    augment_kwargs_list = kwargs.pop("aug", None)
    if augment_kwargs_list is None:
        aug = None
    else:
        aug = build_augmentation(augment_kwargs_list) if is_train else None
    return ClsTransform(img_size=img_size, aug=aug)