import os
import numpy as np
from torchvision import transforms
from PIL import Image


def prepare_data(data_dir, output_dir):
    """
    预处理数据，将医学图像转换为模型可用的格式。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1)
    ])

    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        try:
            img = Image.open(img_path)
            img = transform(img)
            img.save(os.path.join(output_dir, img_name))
        except Exception as e:
            print(f"无法处理图像 {img_name}: {e}")


if __name__ == "__main__":
    data_dir = 'E:\\VAEMIG\\datasets\\Images'  # 原始数据路径
    output_dir = 'E:\\VAEMIG\\datasets\\processed_data'  # 预处理后数据保存路径
    prepare_data(data_dir, output_dir)
