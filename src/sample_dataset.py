import sys
sys.path.append('./')  # NOQA

import os
import numpy as np
import fire
import torch
import PIL
from tqdm import tqdm

from pipeline_emb import EmbPipeline, EmbModel
from dataset import get_dataset

import medmnist
from medmnist import INFO

import ml_collections
import tensorflow as tf
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


def load_pipe(model_id, emb, emb_ckpt, avg_uncond_embeddings=False, dtype=torch.float16):
    checkpoint = torch.load(emb_ckpt)
    emb.load_state_dict(checkpoint)
    emb.to(dtype)
    new_pipe = EmbPipeline.from_pretrained(
        model_id,
        emb=emb,
        safety_checker=None,
        torch_dtype=dtype,
        avg_uncond_embeddings=avg_uncond_embeddings
    )
    return new_pipe

def interpolation_sample(dataset_name, model_root_dir, outdir='inversion_data/cifar10', dm_name="CompVis/stable-diffusion-v1-4",
                         group_id=0, emb_ch=768, num_emb=1000, num_tokens=5, num_classes=10, emb_noise=0.0, interpolation_strength=0.1, num_samples=5,
                        sampling_resolution=128, interpolation='bicubic', save_resolution=32, num_inference_steps=200, train_steps=2000, guidance_scale=2.0,
                        batch_size=100, seed=0):

    root_name = f'{outdir}/res{save_resolution}_{interpolation}'
    
    emb = EmbModel(emb_ch=emb_ch, num_emb=num_emb, num_tokens=num_tokens)
    if dataset_name == 'imagenet':
        import torchvision.transforms as transforms
        from torchvision.transforms import InterpolationMode
        from torchvision.datasets import ImageFolder
        
        TORCH_INTERPOLATION = {"linear": InterpolationMode.BILINEAR,
                                "bilinear": InterpolationMode.BILINEAR,
                                "bicubic": InterpolationMode.BICUBIC,
                                "lanczos": InterpolationMode.LANCZOS,
                                "nearest": InterpolationMode.NEAREST}
        
        imagenet_path = '~/tensorflow_datasets/imagenet/train'
        num_classes=1000
        class_map = {i: [] for i in range(num_classes)}
        transform = transforms.Compose([
            transforms.Resize(sampling_resolution, interpolation=TORCH_INTERPOLATION[interpolation], antialias=True),
            transforms.CenterCrop(sampling_resolution),
            # transforms.ToTensor(),
        ])

        train_dataset = ImageFolder(imagenet_path, transform=transform)
        
        for i, y in enumerate(train_dataset.targets):
            class_map[y].append(i)
        print({i: len(class_map[i]) for i in range(num_classes)})
        
        y_t = []
        for i in range(num_classes):
            y_t.append(i)
            
        emb_path = os.path.join(
            model_root_dir, f'group{group_id}', f'learned_embeds-steps-{train_steps*20}.bin')
    
    elif dataset_name in ['pathmnist', 'dermamnist', 'bloodmnist']:
        info = INFO[dataset_name]
        num_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        train_ds = DataClass(split='train', transform=None, download=True)
        x_train, y_train = train_ds.imgs, train_ds.labels
        y_train = y_train.squeeze()
        # Subset of data
        group_size = num_emb
        cls_idx = np.where(y_train == group_id)[0][0:group_size]
        x_t = x_train[cls_idx]
        y_t = y_train[cls_idx]
        emb_path = os.path.join(
            model_root_dir, f'group{group_id}', f'learned_embeds-steps-{train_steps*5}.bin')
    else:
        # Get original dataset
        config = D(
            name=dataset_name,
            data_dir='~/tensorflow_datasets',
        )
        _, y_train = get_dataset(config, return_raw=True,
                                resolution=save_resolution, train_only=True)

        y_t = y_train[group_id*num_emb:(group_id+1)*num_emb]
        emb_path = os.path.join(
            model_root_dir, f'group{group_id}', f'learned_embeds-steps-{train_steps*2}.bin')
    
    
    if dataset_name in ['pathmnist', 'dermamnist', 'bloodmnist']:
        pipe = load_pipe(dm_name, emb, emb_path, avg_uncond_embeddings=True).to("cuda")
    
        g_cuda = torch.Generator(device='cuda')
        # g_cuda.manual_seed(seed)
        for interpolation_strength in tqdm([0.0, 0.05, 0.1]):
            config_name = f'tstep{train_steps}_infstep{num_inference_steps}_gs{guidance_scale}_noise{emb_noise}_itep{interpolation_strength}_seed{seed}'
            print(f'>> Generate data for {root_name}/{config_name}')
            
            if not os.path.exists(f'{root_name}/class_{group_id:03d}/{config_name}'):
                os.makedirs(
                        f'{root_name}/class_{group_id:03d}/{config_name}', exist_ok=True)
            
            prompt = []
            name_list = []
            for i in range(len(cls_idx)):
                sample_idx = np.random.choice([j for j in range(len(cls_idx)) if j!=i], 
                                            size=num_samples, replace=True).tolist()
                for idx in sample_idx:
                    prompt.append(np.eye(num_emb)[
                                i] + interpolation_strength * (np.eye(num_emb)[idx] - np.eye(num_emb)[i]))
                    name_list.append(
                        f'sample{cls_idx[i]:07d}_{cls_idx[idx]:07d}')
            
            num_batches = len(prompt)//batch_size if len(prompt) % batch_size == 0 else len(prompt)//batch_size+1
            print(f'>> Number of batches: {num_batches}. Number of data: {len(prompt)}.')
            
            for i in range(num_batches):
                p = np.array(prompt[i*batch_size:(i+1)*batch_size])
                name = name_list[i*batch_size:(i+1)*batch_size]
                images = pipe(p, height=sampling_resolution, width=sampling_resolution,
                            num_inference_steps=num_inference_steps, generator=g_cuda, emb_noise=emb_noise,
                            guidance_scale=guidance_scale, eta=1.).images
                
                for idx, img in enumerate(images):
                    img = img.resize((save_resolution, save_resolution),
                                    resample=PIL_INTERPOLATION[interpolation])
                    img.save(
                        f'{root_name}/class_{group_id:03d}/{config_name}/{name[idx]}.png')
    else:
        cls_mapping = {i: [] for i in range(num_classes)}
        for i in range(len(y_t)):
            cls_mapping[y_t[i]].append(i)
        all_ids = list(range(num_emb))
        pipe = load_pipe(dm_name, emb, emb_path, avg_uncond_embeddings=True).to("cuda")
        
        g_cuda = torch.Generator(device='cuda')
        g_cuda.manual_seed(seed)
        
        config_name = f'tstep{train_steps}_infstep{num_inference_steps}_gs{guidance_scale}_noise{emb_noise}_itep{interpolation_strength}_seed{seed}'
        print(f'>> Generate data for {root_name}/{config_name}')
        for i in range(num_classes):
            os.makedirs(f'{root_name}/class_{i:03d}/{config_name}', exist_ok=True)
        prompt = []
        label_list = []
        name_list = []
        for i in range(len(y_t)):
            lb = y_t[i]
            in_class_candidates = [j for j in cls_mapping[lb] if j != i]
            if len(in_class_candidates)>=num_samples:
                sample_idx = np.random.choice(
                    cls_mapping[lb], size=num_samples, replace=False).tolist()
            else:
                out_class_candidates = [j for j in all_ids if j not in cls_mapping[lb]]
                sample_idx = in_class_candidates + np.random.choice(
                    out_class_candidates, size=num_samples-len(in_class_candidates), replace=False).tolist()
            for idx in sample_idx:
                label_list.append(lb)
                prompt.append(np.eye(num_emb)[
                            i] + interpolation_strength * (np.eye(num_emb)[idx] - np.eye(num_emb)[i]))
                name_list.append(
                    f'sample{group_id*num_emb + i:07d}_{group_id*num_emb + idx:07d}')
        
        num_batches = len(prompt)//batch_size if len(prompt) % batch_size == 0 else len(prompt)//batch_size+1
        print(f'>> Number of batches: {num_batches}. Number of data: {len(prompt)}.')
        
        for i in tqdm(range(num_batches)):
            p = np.array(prompt[i*batch_size:(i+1)*batch_size])
            lb = label_list[i*batch_size:(i+1)*batch_size]
            name = name_list[i*batch_size:(i+1)*batch_size]
            images = pipe(p, height=sampling_resolution, width=sampling_resolution,
                        num_inference_steps=num_inference_steps, generator=g_cuda, emb_noise=emb_noise,
                        guidance_scale=guidance_scale, eta=1.).images
            
            for idx, img in enumerate(images):
                img = img.resize((save_resolution, save_resolution),
                                resample=PIL_INTERPOLATION[interpolation])
                img.save(
                    f'{root_name}/class_{lb[idx]:03d}/{config_name}/group{group_id:03d}_{name[idx]}.png')
    
            
if __name__ == "__main__":
    fire.Fire(interpolation_sample)
   