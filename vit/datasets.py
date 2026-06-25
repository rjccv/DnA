import errno
import os
import time

from PIL import Image
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def safe_pil_loader(path, retries=50, delay=0.1, backoff=2.0, max_delay=5.0):
    last_err = None
    sleep_s = delay
    for attempt in range(retries):
        try:
            with open(path, "rb") as handle:
                image = Image.open(handle)
                image.load()
                return image.convert("RGB")
        except OSError as err:
            last_err = err
            if getattr(err, "errno", None) in (errno.EIO, errno.ESTALE, errno.EAGAIN, errno.ETIMEDOUT):
                if attempt < retries - 1:
                    time.sleep(sleep_s)
                    sleep_s = min(max_delay, sleep_s * backoff)
                    continue
            raise
    raise last_err


def build_dataset(is_train, args):
    if args.data_set != "IMNET":
        raise ValueError(f"Unsupported dataset '{args.data_set}'. This repo is now trimmed to the ADeIT ImageNet path.")

    transform = build_transform(is_train=is_train, args=args)
    root = os.path.join(args.data_path, "train" if is_train else "validation")
    dataset = datasets.ImageFolder(root, transform=transform, loader=safe_pil_loader)
    return dataset, 1000


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    transform_list = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        transform_list.append(transforms.Resize(size, interpolation=3))
        transform_list.append(transforms.CenterCrop(args.input_size))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(transform_list)
