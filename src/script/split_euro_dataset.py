import fire

import os
import sys
import json
import shutil

def main():
    input_dir = "$HOME/tensorflow_datasets/downloads/manual/eurosat/2750"
    output_dir = "$HOME/tensorflow_datasets/eurosat"
    
    with open("$HOME/Project/diffusion_inversion/src/split_zhou_EuroSAT.json") as f:
        data = json.load(f)
    
    for k, v in data.items():
        print(k, len(v))
        if not os.path.exists(output_dir):
            os.makedirs('{}/{}'.format(output_dir, k))
    
    for k, v in data.items():
        for i in v:
            if not os.path.exists(os.path.join(output_dir, k, i[0].split('/')[0])):
                os.makedirs(os.path.join(output_dir, k, i[0].split('/')[0]))
            # print(os.path.join(input_dir, i[0]), os.path.join(output_dir, k, i[0]))
            shutil.copy(os.path.join(input_dir, i[0]), os.path.join(output_dir, k, i[0]))

if __name__ == '__main__':
    fire.Fire(main)