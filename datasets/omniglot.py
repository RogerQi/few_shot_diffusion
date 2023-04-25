import os
import argparse

from tqdm import tqdm, trange
from PIL import Image

import torch
import torchvision

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", help="new image size", type=int, default=64)
    parser.add_argument("--out_dir", help="path to output directory", type=str, default="/data")
    args = parser.parse_args()

    data_dir = os.path.join(args.out_dir, "omniglot")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    # Create omniglot dataset from torchvision
    train_ds = torchvision.datasets.Omniglot(root=args.out_dir, download=True, background=True)
    val_ds = torchvision.datasets.Omniglot(root=args.out_dir, download=True, background=False)

    train_ds_path = os.path.join(data_dir, "train_diffusion")
    val_ds_path = os.path.join(data_dir, "val_diffusion")

    if not os.path.exists(train_ds_path):
        os.makedirs(train_ds_path, exist_ok=True)
    
    if not os.path.exists(val_ds_path):
        os.makedirs(val_ds_path, exist_ok=True)

    for i in trange(len(train_ds)):
        img, cls_id = train_ds[i]
        img = img.resize((args.image_size, args.image_size))
        img.save(os.path.join(train_ds_path, "{}_{}.png".format(str(cls_id).zfill(3), str(i).zfill(6))))
    
    for i in trange(len(val_ds)):
        img, cls_id = val_ds[i]
        img = img.resize((args.image_size, args.image_size))
        img.save(os.path.join(val_ds_path, "{}_{}.png".format(str(cls_id).zfill(3), str(i).zfill(6))))

if __name__ == "__main__":
    main()

