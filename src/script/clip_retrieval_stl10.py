
import os
import sys
import json
import fire
import numpy as np
import pandas as pd
import tqdm
from clip_retrieval.clip_client import ClipClient

def get_urls(img_folder, class_idx, k=45):
    # for each image, get the top 45 urls
    print(f'Number of images per query: {k}')
    client = ClipClient(url="https://knn.laion.ai/knn-service",
                    indice_name="laion5B-L-14", num_images=k,
                    use_safety_model=False, use_violence_detector=False, deduplicate=False)
    
    url_map = {}
    for name in tqdm.tqdm(os.listdir(f'{img_folder}/class_{class_idx:03d}')):
        img_path = f'{img_folder}/class_{class_idx:03d}/{name}'
        assert os.path.exists(img_path)
        try:
            results = client.query(image=img_path)
            url_map[name] = {'url': [ele['url'] for ele in results], 'id': [ele['id'] for ele in results]}
        except:
            print('Error: Class{} {}'.format(class_idx, name))
        # print(url_map[name])
    return url_map


def url_analysis(url_map, key='id'):
    # count the unique urls
    unique_urls = set()
    
    for name, ele in url_map.items():
        for url in ele[key]:
            unique_urls.add(url)
            
    print('Unique urls:')
    print(len(unique_urls))
    
    # count occurrences of each url
    occurrences = {}
    for url in unique_urls:
        count = 0
        for name, ele in url_map.items():
            if url in ele[key]:
                count += 1
        occurrences[url] = count
    # print('Occurrences:')
    # print(occurrences)
    
    # compute the average number of occurrences
    mean = sum(occurrences.values()) / len(occurrences)
    print('Mean occurrences:')
    print(mean)
    # The underlying reason is because the clip embedding does not capture the discriminative information of the image on the target domain.


def main(class_idx, k=45):
    img_folder = "/ssd005/projects/diffusion_inversion/real_data/stl10/scaling/res96_bicubic"
    output_dir = f'clip_retrieval/stl10'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # url_map = get_urls(img_folder, class_idx=class_idx, k=k)

    # with open(f"{output_dir}/cls{class_idx}_k{k}.json", "w") as f:
    #     json.dump(url_map, f)
    
    with open(f"{output_dir}/cls{class_idx}_k{k}.json") as f:
        url_map = json.load(f)
    
    url_analysis(url_map)
    
    unique_urls = set()
    res = []

    for name, ele in url_map.items():
        for url in ele['url']:
            if url not in unique_urls:
                res.append({'url': url})
            unique_urls.add(url)

    print('Unique urls:')
    print(len(unique_urls))

    with open(f"{output_dir}/cls{class_idx}_k{k}_unique.json", mode="w") as f:
        json.dump(res, f)
    
    # image_folder = f"/ssd005/projects/diffusion_inversion/inversion_data/stl10/knn_retrieval/k{k}/class_{class_idx:03d}"
    # os.system(
    #     f'img2dataset --input_format=json --url_list={output_dir}/cls{class_idx}_k{k}_unique.json --output_folder={image_folder} --thread_count=64 --image_size=96 --output_format=files')
    
    image_folder = f"/ssd005/projects/diffusion_inversion/inversion_data/stl10/knn_retrieval/k{k}_unresize/class_{class_idx:03d}"
    os.system(
        f'img2dataset --input_format=json --url_list={output_dir}/cls{class_idx}_k{k}_unique.json --output_folder={image_folder} --thread_count=64 --output_format=files')

if __name__ == "__main__":
    fire.Fire(main)
    # python clip_retrieval_stl10.py --class_idx=0 --k=10
    # img2dataset --input_format=json --url_list=urls_10.json --output_folder=/ssd005/projects/diffusion_inversion/inversion_data/stl10/knn_retrieval/ --thread_count=64 --image_size=96 --output_format=files
    
    # https://github.com/rom1504/img2dataset
    