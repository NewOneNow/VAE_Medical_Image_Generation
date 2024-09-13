import os
from torch.utils.data import Dataset
from PIL import Image


class MedicalImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super(MedicalImageDataset, self).__init__()
        self.data_dir = data_dir
        self.image_names = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('L')  # 单通道灰度图
        if self.transform:
            image = self.transform(image)
        return image
