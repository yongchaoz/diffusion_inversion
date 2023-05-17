import sys
import argparse
import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import tensorflow_datasets as tfds

import datasets
import diffusers
import PIL
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import medmnist
from medmnist import INFO

####### Modification (START) #######

import ml_collections
import tensorflow as tf
sys.path.append('.')  # NOQA
from pipeline_emb import EmbPipeline, EmbModel
from dataset import get_dataset
tf.config.experimental.set_visible_devices([], "GPU")


def D(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)
####### Modification (END) #######


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------
logger = get_logger(__name__)


def np_tile_imgs(imgs, *, pad_pixels=1, pad_val=255, num_col=0):
    """NumPy utility: tile a batch of images into a single image.

    Args:
      imgs: np.ndarray: a uint8 array of images of shape [n, h, w, c]
      pad_pixels: int: number of pixels of padding to add around each image
      pad_val: int: padding value
      num_col: int: number of columns in the tiling; defaults to a square

    Returns:
      np.ndarray: one tiled image: a uint8 array of shape [H, W, c]
    """
    if pad_pixels < 0:
        raise ValueError('Expected pad_pixels >= 0')
    if not 0 <= pad_val <= 255:
        raise ValueError('Expected pad_val in [0, 255]')

    imgs = np.asarray(imgs)
    if imgs.dtype != np.uint8:
        raise ValueError('Expected uint8 input')
    # if imgs.ndim == 3:
    #   imgs = imgs[..., None]
    n, h, w, c = imgs.shape
    if c not in [1, 3]:
        raise ValueError('Expected 1 or 3 channels')

    if num_col <= 0:
        # Make a square
        ceil_sqrt_n = int(np.ceil(np.sqrt(float(n))))
        num_row = ceil_sqrt_n
        num_col = ceil_sqrt_n
    else:
        # Make a B/num_per_row x num_per_row grid
        assert n % num_col == 0
        num_row = int(np.ceil(n / num_col))

    imgs = np.pad(
        imgs,
        pad_width=((0, num_row * num_col - n), (pad_pixels, pad_pixels),
                   (pad_pixels, pad_pixels), (0, 0)),
        mode='constant',
        constant_values=pad_val
    )
    h, w = h + 2 * pad_pixels, w + 2 * pad_pixels
    imgs = imgs.reshape(num_row, num_col, h, w, c)
    imgs = imgs.transpose(0, 2, 1, 3, 4)
    imgs = imgs.reshape(num_row * h, num_col * w, c)

    if pad_pixels > 0:
        imgs = imgs[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels, :]
    if c == 1:
        imgs = imgs[Ellipsis, 0]
    return imgs


def save_tiled_imgs(filename, imgs, pad_pixels=1, pad_val=255, num_col=0):
    PIL.Image.fromarray(
        np_tile_imgs(
            imgs, pad_pixels=pad_pixels, pad_val=pad_val,
            num_col=num_col)).save(filename)


def save_progress(emb_model, accelerator, save_path):
    logger.info("Saving embeddings")
    model = accelerator.unwrap_model(emb_model)
    learned_embeds = model.emb.weight
    learned_embeds_dict = {
        'emb.weight': learned_embeds.detach().cpu(),
    }
    torch.save(learned_embeds_dict, save_path)


def save_image(pipe, image_dir, step, num_emb, device, resolution):
    prompt = list(range(25))
    pipe.to(device)
    # for guidance_scale in [2.0, 4.0]:
    for guidance_scale in [2.0]:
        filename = os.path.join(
            image_dir, f'{step:05d}_gs{guidance_scale}.jpg')
        logger.info(f"Saving images to {filename}")
        images = pipe(np.eye(num_emb)[prompt], height=resolution, width=resolution,
                      num_inference_steps=50, guidance_scale=guidance_scale, eta=1., generator=None).images
        images = np.stack([np.array(x) for x in images])
        save_tiled_imgs(filename, images)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    #### New arguments (START) ####
    parser.add_argument("--num_emb", type=int, default=10)
    parser.add_argument("--num_tokens", type=int, default=3)
    parser.add_argument("--group_id", type=int, default=None)
    parser.add_argument("--interpolation", type=str,
                        default='bilinear', help='bilinear or bicubic')
    parser.add_argument("--dataset_name", type=str,
                        default='cifar10')
    parser.add_argument("--data_dir", type=str,
                        default='~/tensorflow_datasets')
    #### New arguments (END) ####

    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_image_steps",
        type=int,
        default=50,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--only_save_embeds",
        action="store_true",
        default=False,
        help="Save only the embeddings for the new concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    # parser.add_argument(
    #     "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    # )
    parser.add_argument("--repeats", type=int, default=100,
                        help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float,
                        default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                        help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def load_data(ds, batch_size=1000):
    ds = ds.batch(batch_size=batch_size)
    x_list = []
    y_list = []
    for x, y in tfds.as_numpy(ds):
        x_list.append(x)
        y_list.append(y)
    return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)


class EmbDataset(Dataset):
    def __init__(self, dataset_name, data_dir, 
                 group_id=None, num_emb=1000, size=512,
                 interpolation='bicubic', center_crop=False, hflip=False):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.group_id = group_id
        self.num_emb = num_emb
        self.size = size
        self.interpolation = PIL_INTERPOLATION[interpolation]
        self.center_crop = center_crop
        self.hflip = hflip

        if dataset_name == 'imagenet':
            import torchvision.transforms as transforms
            from torchvision.transforms import InterpolationMode
            from torchvision.datasets import ImageFolder
            
            TORCH_INTERPOLATION = {"linear": InterpolationMode.BILINEAR,
                                   "bilinear": InterpolationMode.BILINEAR,
                                   "bicubic": InterpolationMode.BICUBIC,
                                   "lanczos": InterpolationMode.LANCZOS,
                                   "nearest": InterpolationMode.NEAREST}
            
            imagenet_path = '$HOME/tensorflow_datasets/imagenet/train'
            num_classes=1000
            class_map = {i: [] for i in range(num_classes)}
            transform = transforms.Compose([
                transforms.Resize(size, interpolation=TORCH_INTERPOLATION[interpolation], antialias=True),
                transforms.CenterCrop(size),
                # transforms.ToTensor(),
            ])

            train_dataset = ImageFolder(imagenet_path, transform=transform)
            
            for i, y in enumerate(train_dataset.targets):
                class_map[y].append(i)
            print({i: len(class_map[i]) for i in range(num_classes)})
            
            x_t = []
            y_t = []
            for i in range(num_classes):
                x_t.append(train_dataset[class_map[i][group_id]][0])
                y_t.append(i)
        elif dataset_name == 'eurosat':
            import torchvision.transforms as transforms
            from torchvision.transforms import InterpolationMode
            from torchvision.datasets import ImageFolder

            TORCH_INTERPOLATION = {"linear": InterpolationMode.BILINEAR,
                                   "bilinear": InterpolationMode.BILINEAR,
                                   "bicubic": InterpolationMode.BICUBIC,
                                   "lanczos": InterpolationMode.LANCZOS,
                                   "nearest": InterpolationMode.NEAREST}

            imagenet_path = '$HOME/tensorflow_datasets/eurosat/train'
            num_classes = 10
            class_map = {i: [] for i in range(num_classes)}
            transform = transforms.Compose([
                transforms.Resize(
                    size, interpolation=TORCH_INTERPOLATION[interpolation], antialias=True),
                transforms.CenterCrop(size),
                # transforms.ToTensor(),
            ])

            train_dataset = ImageFolder(imagenet_path, transform=transform)
            
            random_idx = [451, 300, 209, 245, 294, 355, 91, 185, 80, 456, 74, 476, 167, 134, 380, 496, 78, 337, 296, 127, 97, 99, 411, 369, 297][:num_emb]

            for i, y in enumerate(train_dataset.targets):
                class_map[y].append(i)
            print({i: len(class_map[i]) for i in range(num_classes)})
            
            data_idx = [class_map[group_id][idx] for idx in random_idx]
            x_t = [train_dataset[idx][0] for idx in data_idx]
            y_t = [group_id for _ in range(len(x_t))]
                
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
        else:
            config = D(
                name=dataset_name,
                data_dir=data_dir
            )

            x_train, y_train = get_dataset(
                config, return_raw=True, resolution=size, train_only=True)

            if group_id is not None:
                x_t = x_train[group_id*num_emb:(group_id+1)*num_emb]
                y_t = y_train[group_id*num_emb:(group_id+1)*num_emb]
            else:
                cls_idx = {}
                for i in range(config.num_classes):
                    cls_idx[i] = np.where(y_train == i)[
                        0][0:num_emb // config.num_classes]
                x_t = np.concatenate([x_train[idx]
                                    for k, idx in cls_idx.items()], axis=0)
                y_t = np.concatenate([y_train[idx]
                                    for k, idx in cls_idx.items()], axis=0)
        idx_t = np.arange(len(x_t))

        self.x_t = x_t
        self.y_t = y_t
        self.idx_t = idx_t

    def __len__(self):
        if self.dataset_name in ['imagenet', 'eurosat']:
            return len(self.x_t)
        else:
            return self.x_t.shape[0]

    def __getitem__(self, idx):
        example = {}
        if self.dataset_name in ['imagenet', 'eurosat']:
            image = self.x_t[idx]
        else:
            img = self.x_t[idx].astype(np.uint8)
            image = Image.fromarray(img)
            image = image.resize((self.size, self.size),
                                resample=self.interpolation)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)

        example["pixel_values"] = image
        example["label"] = self.y_t[idx]
        example["emb_id"] = np.eye(self.num_emb)[
            self.idx_t[idx]].astype(np.float32)
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            args.output_dir = os.path.join(
                args.output_dir, f'res{args.resolution}_{args.interpolation}/emb{args.num_emb}_token{args.num_tokens}_lr{args.learning_rate}_{args.lr_scheduler}/group{args.group_id}')
            os.makedirs(args.output_dir, exist_ok=True)

    print(f"Output directory: {args.output_dir}")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    emb_ch = text_encoder.config.hidden_size
    emb_model = EmbModel(emb_ch=emb_ch, num_emb=args.num_emb,
                         num_tokens=args.num_tokens)

    train_dataset = EmbDataset(dataset_name=args.dataset_name,
                               data_dir=args.data_dir,
                               group_id=args.group_id,
                               num_emb=args.num_emb,
                               size=args.resolution,
                               interpolation=args.interpolation)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=False)

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        # text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        emb_model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    emb_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        emb_model, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("diffusion_emb", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    print(f"  Instantaneous batch size per device = {args.train_batch_size}")

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        try:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1]
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = resume_global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % num_update_steps_per_epoch
        except:
            print('Training from scratch!')
            args.resume_from_checkpoint = False

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    print('>>> Save training images')
    images = np.stack([np.array(x) for x in train_dataset.x_t[:100]])
    save_tiled_imgs(f'{args.output_dir}/original.jpg', images)
    
    for epoch in range(first_epoch, args.num_train_epochs):
        # text_encoder.train()
        emb_model.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            # with accelerator.accumulate(text_encoder):
            with accelerator.accumulate(emb_model):
                # Convert images to latent space
                if 'latent' in batch.keys():
                    latents = batch['latent']
                else:
                    latents = vae.encode(batch["pixel_values"].to(
                        dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps)

                # Get the conditioning vector
                encoder_hidden_states = emb_model(batch['emb_id'])

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps,
                                  encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(),
                                  target.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(
                        args.output_dir, f"learned_embeds-steps-{global_step}.bin")
                    save_progress(emb_model, accelerator, save_path)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if global_step % args.save_image_steps == 0:
                    pipeline = EmbPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        emb=emb_model,
                        safety_checker=None,
                        torch_dtype=weight_dtype,
                        avg_uncond_embeddings=True,
                    )
                    save_image(pipeline, args.output_dir, global_step,
                               args.num_emb, accelerator.device, args.resolution)
                    del pipeline
                    
            logs = {"loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "learned_embeds.bin")
        save_progress(emb_model, accelerator, save_path)

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training",
                             blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
