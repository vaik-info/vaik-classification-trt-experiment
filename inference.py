import argparse
import os
import glob
import json
import tqdm
import numpy as np
from PIL import Image
from tqdm import tqdm
from vaik_classification_trt_inference.trt_model import TrtModel


def main(input_saved_model_dir_path, input_classes_path, input_image_dir_path, output_json_dir_path):
    os.makedirs(output_json_dir_path, exist_ok=True)
    with open(input_classes_path, 'r') as f:
        classes = f.readlines()
    classes = tuple([label.strip() for label in classes])

    model = TrtModel(input_saved_model_dir_path, classes)

    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_path_list = []
    for files in types:
        image_path_list.extend(glob.glob(os.path.join(input_image_dir_path, '*', files), recursive=True))
    image_list = []
    for image_path in tqdm(image_path_list, 'read images'):
        image = np.asarray(Image.open(image_path).convert('RGB'))
        image_list.append(image)
    import time
    start = time.time()
    output, raw_pred = model.inference(image_list)
    end = time.time()

    for image_path, output_elem in zip(image_path_list, output):
        output_json_path = os.path.join(output_json_dir_path, os.path.splitext(os.path.basename(image_path))[0]+'.json')
        output_elem['answer'] = os.path.basename(os.path.dirname(image_path))
        output_elem['image_path'] = image_path
        with open(output_json_path, 'w') as f:
            json.dump(output_elem, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    print(f'{len(image_list)/(end-start)}[images/sec]')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--input_saved_model_dir_path', type=str, default='~/output_trt_model/model.fp16.trt')
    parser.add_argument('--input_classes_path', type=str, default='~/.vaik-mnist-classification-dataset/classes.txt')
    parser.add_argument('--input_image_dir_path', type=str, default='~/.vaik-mnist-classification-dataset/valid')
    parser.add_argument('--output_json_dir_path', type=str, default='~/.vaik-mnist-classification-dataset/valid_inference')
    args = parser.parse_args()

    args.input_saved_model_dir_path = os.path.expanduser(args.input_saved_model_dir_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)
    args.input_image_dir_path = os.path.expanduser(args.input_image_dir_path)
    args.output_json_dir_path = os.path.expanduser(args.output_json_dir_path)

    main(**args.__dict__)