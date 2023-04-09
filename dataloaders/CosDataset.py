import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# NOTE: you need to download the mapillary_sls dataset from  https://github.com/FrederikWarburg/mapillary_sls
# make sure the path where the mapillary_sls validation dataset resides on your computer is correct.
# the folder named train_val should reside in DATASET_ROOT path (that's the only folder you need from mapillary_sls)
# I hardcoded the groundtruth for image to image evaluation, otherwise it would take ages to run the groundtruth script at each epoch.
DATASET_ROOT = 'datasets/cos_val/'

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception('Please make sure the path to mapillary_sls dataset is correct')

if not path_obj.joinpath('train_val'):
    raise Exception(f'Please make sure the directory train_val from mapillary_sls dataset is situated in the directory {DATASET_ROOT}')

class COS(Dataset):
    def __init__(self, input_transform = None):
        
        self.input_transform = input_transform
        test_file = os.path.join(DATASET_ROOT, 'test.json')
        with open(test_file, 'r') as f:
            json_data = json.load(f)
        test_data = []
        for items in json_data.values():
            test_data.extend(items)
        image_2_label = {}
        for item in test_data:
            image = item['image']
            image_name = image.split('/')[-1]
            label = item['label']
            image_2_label[image_name] = label
        brands = os.listdir(os.path.join(DATASET_ROOT, 'database'))
        self.dbImages = []
        for brand in brands:
            brand_path = os.path.join(DATASET_ROOT, 'database', brand)
            for img in os.listdir(brand_path):
                self.dbImages.append(os.path.join(brand_path, img))
        query_brands = os.listdir(os.path.join(DATASET_ROOT, 'query'))
        # hard coded query image names.
        self.qImages = []
        for brand in query_brands:
            brand_path = os.path.join(DATASET_ROOT, 'query', brand)
            for img in os.listdir(brand_path):
                self.qImages.append(os.path.join(brand_path, img))
        self.qIdx = [i for i in range(len(self.qImages))]
        # hard coded groundtruth (correspondence between each query and its matches)
        self.pIdx = []
        for one in self.qImages:
            image_name = one.split('/')[-1]
            label = image_2_label[image_name]
            self.pIdx.append(label)
        # concatenate reference images then query images so that we can use only one dataloader
        self.images = np.concatenate((self.dbImages, self.qImages))
        
        # we need to keeo the number of references so that we can split references-queries 
        # when calculating recall@K
        self.num_references = len(self.dbImages)
    
    def __getitem__(self, index):
        img = Image.open(self.images[index])
        # 变成3通道
        img = img.convert("RGB")

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)