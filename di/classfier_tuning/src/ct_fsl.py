import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import wandb
from src.models.finetune import finetune_fsl
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip
from PIL import Image

def _convert_to_rgb(image):
    return image.convert('RGB')

normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

def classifier_tuning(args):
    assert args.save is not None, 'Please provide a path to store models'
    print('import success')
    
    classification_head_path = os.path.join(args.save, f'head.pt')
    args.data_location_real = os.path.join(args.data_location_real, f'shot_{args.shot}')
    args.data_location_syn = os.path.join(args.data_location_syn, f'shot_{args.shot}')
    args.save = os.path.join(args.save, f'shot_{args.shot}', args.cache_name_syn)
    args.cache_dir = os.path.join(args.cache_dir, f'shot_{args.shot}')
    args.results_db = os.path.join(args.results_db, f'shot_{args.shot}', args.cache_name_syn)
    
    # Build and save zero-shot model
    image_encoder = ImageEncoder(args, keep_lang=True)
    if not os.path.exists(classification_head_path):
        classification_head = get_zeroshot_classifier(args, image_encoder.model)
        classification_head.save(classification_head_path)
    else:
        classification_head = ClassificationHead.load(classification_head_path)
    delattr(image_encoder.model, 'transformer')
    classifier = ImageClassifier(image_encoder, classification_head, process_images=False)

    zeroshot_checkpoint = os.path.join(args.save, 'zeroshot'+args.train_dataset+'.pt')
    classifier.save(zeroshot_checkpoint)

    # Standard fine-tuning
    args.load = zeroshot_checkpoint
    args.save = os.path.join(args.save, 'finetuned')

    # Mimic eurosat low-res images, val data aug
    train_data_aug = Compose([
        # Resize(64), # resize to 32/64 for Cifar / Eurosat
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])
    
    wandb.init(project=f"eurosat-fewshot", config=args)
    finetuned_checkpoint = finetune_fsl(args, train_data_aug)


if __name__ == '__main__':
    args = parse_arguments()
    classifier_tuning(args)
