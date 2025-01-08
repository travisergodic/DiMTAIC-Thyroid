import random

from PIL import Image
from torchvision.transforms import InterpolationMode, ColorJitter
import torchvision.transforms.functional as F

from src.registry import AUGMENT
from src.utils import generate_cutout_bbox


def pad_to_square(image, mask=None):
    W, H = image.size
    if W > H:
        pad_size = (W - H) // 2
        padding = (0, pad_size, 0, pad_size)
    else:
        pad_size = (H - W) // 2
        padding = (pad_size, 0, pad_size, 0)
        
    image = F.pad(image, padding=padding, fill=0, padding_mode="constant")
    if mask is not None:
        mask = F.pad(mask, padding=padding, fill=0, padding_mode="constant")
    return image, mask


class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask, label=None, bboxes=None):
        for transform in self.transforms:
            image, mask = transform(image, mask, label, bboxes)
        return image, mask


@AUGMENT.register("color_jitter")
class RandomColorJitter:
    def __init__(self, **kwargs):
        self.jitter = ColorJitter(**kwargs)

    def __call__(self, image, mask, label=None, bboxes=None):
        return self.jitter(image), mask
    

@AUGMENT.register("perspective")
class RandomPerspectiveTransform:
    def __init__(self, distortion_scale, prob):
        self.distortion_scale = distortion_scale
        self.prob = prob

    def __call__(self, image, mask, label=None, bboxes=None):
        if random.random() > self.prob:
            return image, mask
        
        width, height = image.size
        startpoints = [
            (0, 0),  # top-left
            (width - 1, 0),  # top-right
            (0, height - 1),  # bottom-left
            (width - 1, height - 1)  # bottom-right
        ]

        # Apply random shifts to the endpoints
        def random_point_near(x, y, width, height, scale):
            """Generate a random point near (x, y) within a certain percentage scale."""
            return (
                x + random.uniform(-width * scale, width * scale),
                y + random.uniform(-height * scale, height * scale)
            )

        # Generate endpoints by shifting startpoints randomly
        endpoints = [
            random_point_near(0, 0, width, height, self.distortion_scale),  # top-left
            random_point_near(width - 1, 0, width, height, self.distortion_scale),  # top-right
            random_point_near(0, height - 1, width, height, self.distortion_scale),  # bottom-left
            random_point_near(width - 1, height - 1, width, height, self.distortion_scale)  # bottom-right
        ]
        image = F.perspective(image, startpoints, endpoints, interpolation=InterpolationMode.BICUBIC)
        if mask is not None:
            mask = F.perspective(mask, startpoints, endpoints, interpolation=InterpolationMode.NEAREST)
        return image, mask
    

@AUGMENT.register("hflip")
class HFlipTransform:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, mask, label=None, bboxes=None):
        if random.random() < self.prob:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)
        return image, mask


@AUGMENT.register("vflip")
class VFlipTransform:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, mask, label=None, bboxes=None):
        if random.random() < self.prob:
            image = F.vflip(image)
            if mask is not None:
                mask = F.vflip(mask)
        return image, mask


@AUGMENT.register("random_crop")
class RandomCropTransform:
    def __init__(self, crop_ratio):
        self.crop_ratio = crop_ratio

    def __call__(self, image, mask, label=None, bboxes=None):
        W, H = image.size
        if self.crop_ratio < 1:
            w, h = random.randint(int(self.crop_ratio * W), W), random.randint(int(self.crop_ratio * H), H)
            i, j = random.randint(0, H-h), random.randint(0, W-w)
            image = F.crop(image, i, j, h, w)
            if mask is not None:
                mask = F.crop(mask, i, j, h, w)
            image, mask = pad_to_square(image, mask)
        return image, mask
    

@AUGMENT.register("random_cutout")
class RandomCutoutTransform:
    def __init__(self, cutout_prob):
        self.cutout_prob = cutout_prob

    def __call__(self, image, mask, label=None, bboxes=None):
        if random.uniform(0, 1) > self.cutout_prob:
            return image, None
        
        W, H = image.size
        cutout_bboxes = generate_cutout_bbox(label, obj_bboxes=bboxes, image_width=W, image_height=H)
        for cutout_bbox in cutout_bboxes:
            draw = Image.new("RGB", (cutout_bbox[2] - cutout_bbox[0], cutout_bbox[3] - cutout_bbox[1]), (0, 0, 0))
            image.paste(draw, (cutout_bbox[0], cutout_bbox[1]))
        return image, None
    

def build_augmentation(kwargs_list=None):
    if kwargs_list is None:
        return
    
    transform_list = []
    for kwargs in kwargs_list:
        transform_list.append(AUGMENT.build(**kwargs))
    return ComposeTransforms(transform_list)