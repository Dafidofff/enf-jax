import cv2
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils import data
from torchvision.transforms import v2


class ImageNetWithID(data.Dataset):
    """ ImageNet dataset for training a neural field and using it for downstream classification. The labels
        are the ImageNet classes specified by modules. These modules can be translated with the LOC_synset_mapping.txt.
        The images are resized to 224x224 and normalized to the ImageNet mean and std.

        Datastructure:  {root}/ILSVRC/CLS-LOC/train/{class}/{image}.JPEG

        Args:
            data_dir (str): Path to the ImageNet dataset.
            split (str): Split of the dataset. Choose from ['train', 'val'].
            img_size (int): Size of the image to resize to.
    """

    def __init__(self, data_dir, split:str = 'train', img_size:int = 224):
        assert split in ['train', 'val'], f"Split {split} not supported. Choose from ['train', 'val']"

        # Glob downloaded imagenet dir
        data_dir = Path(data_dir)
        self.img_paths = list((data_dir / f'ILSVRC/Data/CLS-LOC/{split}/').glob('**/*.JPEG'))
        label_mappings = data_dir / 'LOC_synset_mapping.txt'

        # Create path to label/module mapping
        self.img_dict = defaultdict(list)
        for img_path in self.img_paths:
            img_module = str(img_path).split('/')[-2]
            self.img_dict[img_module].append(img_path)
        self.img_modules = list(self.img_dict.keys())

        # Define imagenet transforms
        self.T = v2.Compose([
            v2.ToTensor(),
            v2.ToDtype(torch.float32),
            v2.Resize(256),
            v2.CenterCrop(img_size),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.len_dir = len(self.img_paths)
        
    def __len__(self):
        return self.len_dir
    
    def read_img(self, img_dir):
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.T(img)
    
    def __getitem__(self, idx):
        smp_path = self.img_paths[idx]
        smp_img = self.read_img(smp_path)

        # Move channels last
        smp_img = smp_img.permute(1, 2, 0)

        # label 
        pos_module = str(smp_path).split('/')[-2]
        label = self.img_modules.index(pos_module)
        return smp_img, label, idx


if __name__ == '__main__':

    image_dir = '/media/data/imagenet-object-localization-challenge/'
    dataset = ImageNetWithID(image_dir)

    smp = dataset[0]
    print(smp[0].shape)