import os
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# NOTE: you need to download the mapillary_sls dataset from  https://github.com/FrederikWarburg/mapillary_sls
# make sure the path where the mapillary_sls validation dataset resides on your computer is correct.
# the folder named train_val should reside in DATASET_ROOT path (that's the only folder you need from mapillary_sls)
# I hardcoded the groundtruth for image to image evaluation, otherwise it would take ages to run the groundtruth script at each epoch.
DATASET_ROOT = 'datasets/msls_val/'

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception('Please make sure the path to mapillary_sls dataset is correct')

if not path_obj.joinpath('train_val'):
    raise Exception(f'Please make sure the directory train_val from mapillary_sls dataset is situated in the directory {DATASET_ROOT}')

class MSLS(Dataset):
    def __init__(self, input_transform = None):
        
        self.input_transform = input_transform
        
        # hard coded reference image names, this avoids the hassle of listing them at each epoch., 图片列表
        self.dbImages = np.load(os.path.join(path_obj,'msls_val_dbImages.npy'))
        
        # hard coded query image names.， 图片列表
        self.qImages = np.load(os.path.join(path_obj,'msls_val_qImages.npy'))
        
        # hard coded index of query images，
        self.qIdx = np.load(os.path.join(path_obj,'msls_val_qIdx.npy'))
        
        # hard coded groundtruth (correspondence between each query and its matches)， groundtruth，每个query对应的匹配图片label
        self.pIdx = np.load(os.path.join(path_obj,'msls_val_pIdx.npy'), allow_pickle=True)
        
        # 拼接参考图像和查询图像，因此我们可以只用一个dataloader
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))
        
        # we need to keep the number of references so that we can split references-queries
        # when calculating recall@K
        self.num_references = len(self.dbImages)
    
    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT+self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)