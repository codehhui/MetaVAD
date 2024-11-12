import argparse
import os

from PIL import Image
import PIL.ImageOps
import numpy as np

from straug.blur import MotionBlur
from straug.camera import Brightness

def process_image(img_path, opt, ops, rng, file_counter, category_folder):
    img = Image.open(img_path)
    img = img.resize((opt.width, opt.height))

    for op in ops:
        filename = os.path.join(opt.results, category_folder, f"{file_counter:05d}.jpg")
        file_counter += 1
        out_img = op(img, mag=0)  # 使用特定的 mag 值
        if opt.gray:
            out_img = PIL.ImageOps.grayscale(out_img)
        out_img.save(filename)

    return file_counter

def process_folder(folder_path, opt, ops, rng):
    file_counter = 0
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            category_folder = dir_name
            category_path = os.path.join(root, category_folder)
            results_dir = os.path.join(opt.results, category_folder)
            os.makedirs(results_dir, exist_ok=True)  # 创建每个类别的结果文件夹

            image_files = [f for f in os.listdir(category_path) if f.endswith(('png', 'jpg', 'jpeg', 'bmp'))]
            for image_file in image_files:
                image_path = os.path.join(category_path, image_file)
                file_counter = process_image(image_path, opt, ops, rng, file_counter, category_folder)
    return file_counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default="E:\wlh\straug-main\images\frames\abuse", help='Root folder containing images')
    parser.add_argument('--results', default="E:\wlh\straug-main\images\frames\abusezengqiang", help='Root folder for augmented image files')
    parser.add_argument('--gray', action='store_true', help='Convert to grayscale')
    parser.add_argument('--width', default=360, type=int, help='Default image width')
    parser.add_argument('--height', default=240, type=int, help='Default image height')
    parser.add_argument('--seed', default=0, type=int, help='Random number generator seed')
    opt = parser.parse_args()
    os.makedirs(opt.results, exist_ok=True)

    rng = np.random.default_rng(opt.seed)
    ops = [MotionBlur(rng), Brightness(rng)]

    file_count = process_folder(opt.image_folder, opt, ops, rng)
    print(f"Total {file_count} images processed.")
    print('Random token:', rng.integers(2 ** 16))
