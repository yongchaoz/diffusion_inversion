import os
import sys
import fire
import ml_collections
import torch
import tensorflow as tf
import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset

from diffusers import AutoencoderKL

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset import get_dataset  # NOQA

tf.config.experimental.set_visible_devices([], "GPU")


PIL_INTERPOLATION = {
    "linear": PIL.Image.Resampling.BILINEAR,
    "bilinear": PIL.Image.Resampling.BILINEAR,
    "bicubic": PIL.Image.Resampling.BICUBIC,
    "lanczos": PIL.Image.Resampling.LANCZOS,
    "nearest": PIL.Image.Resampling.NEAREST,
}


def D(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


class EmbDataset(Dataset):
    def __init__(self, x, y, size=32, interpolation=Image.BICUBIC):
        self.x = x
        self.y = y
        self.size = size
        self.interpolation = interpolation

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        img = self.x[idx].astype(np.uint8)
        image = Image.fromarray(img)
        image = image.resize((self.size, self.size),
                             resample=self.interpolation)
        image = np.array(image).astype(np.float32)
        image = (image / 127.5 - 1.0)
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image, self.y[idx]


def main(dataset_name, root_name, split='train', group_size=100, batch_size=50, sampling_resolution=128, 
         interpolation='bicubic', save_resolution=32, num_classes=10):

    device = "cuda"
    dataset_config = D(
        name=dataset_name,
        data_dir='~/tensorflow_datasets'
    )
    x_train, y_train, x_test, y_test = get_dataset(
        dataset_config, return_raw=True, train_only=False)

    root_name = os.path.join(root_name, f'{dataset_name}_{split}')
    model_id = "CompVis/stable-diffusion-v1-4"
    ae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)

    config_name = f'{model_id.replace("/", "-")}_sample{sampling_resolution}_{interpolation}_save{save_resolution}'

    for i in range(num_classes):
        os.makedirs(f'{root_name}/class_{i:03d}/{config_name}', exist_ok=True)

    if split == 'train':
        dataset = EmbDataset(x_train, y_train, size=sampling_resolution,
                             interpolation=PIL_INTERPOLATION[interpolation])
    elif split == 'test':
        dataset = EmbDataset(x_test, y_test, size=sampling_resolution,
                             interpolation=PIL_INTERPOLATION[interpolation])
    else:
        raise ValueError(f'Unknown split {split}')

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    for i, (img, lb) in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            latents = ae.encode(img.to(device)).latent_dist.mode()
            image = ae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            images = image.cpu().permute(0, 2, 3, 1).float().numpy()
            images = (images * 255).round().astype("uint8")
            images = [Image.fromarray(image) for image in images]

        for idx, img in enumerate(images):
            if sampling_resolution != save_resolution:
                img = img.resize((save_resolution, save_resolution),
                                 resample=PIL_INTERPOLATION[interpolation])
            img.save(
                f'{root_name}/class_{lb[idx]:03d}/{config_name}/group{(i*batch_size+idx)//group_size:02d}_sample{(i*batch_size+idx):05d}.png')


if __name__ == '__main__':
    fire.Fire(main)
    # python convert_vae.py --dataset_name=cifar10 --root_name=$HOME/inversion_data --batch_size=100 --interpolation=bicubic --save_resolution=128 --num_classes=10 --split=train --sampling_resolution=128
