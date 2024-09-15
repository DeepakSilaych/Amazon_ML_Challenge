import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .utils import download_image

class ProductImageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """Initialize the dataset with the CSV file and image directory."""
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a data sample at the given index."""
        record = self.data.iloc[idx]
        img_name = os.path.join(self.image_dir, f"{record['index']}.jpg")

        if not os.path.exists(img_name):
            download_image(record['image_link'], img_name)

        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        entity_value = record.get('entity_value', '')

        return {
            'index': record['index'],
            'image': image,
            'group_id': record['group_id'],
            'entity_name': record['entity_name'],
            'entity_value': entity_value
        }
