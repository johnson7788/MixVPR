# https://github.com/amaralibey/gsv-cities
import os
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# NOTE: Hard coded path to dataset folder 
BASE_PATH = 'datasets/comestic/'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to gsv_cities')

class ComesticDataset(Dataset):
    def __init__(self,
                 brands=['倩碧', '纨素之肤'],
                 img_per_product=4,
                 min_img_per_product=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=BASE_PATH
                 ):
        super(ComesticDataset, self).__init__()
        self.base_path = base_path
        self.brands = brands

        # assert img_per_product <= min_img_per_product, \
        #     f"img_per_product should be less than {min_img_per_product}"
        self.img_per_product = img_per_product
        self.min_img_per_product = min_img_per_product
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        
        # generate the dataframe contraining images metadata
        self.dataframe = self.__getdataframes()
        
        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)
        
    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        train_file = os.path.join(self.base_path, 'train.json')
        # 读取成dataframe
        with open(train_file, 'r') as f:
            json_data = json.load(f)
        train_data = []
        for value in json_data.values():
            train_data.extend(value)
        df = pd.DataFrame(train_data)
        # keep only places depicted by at least min_img_per_product images
        res = df[df.groupby('label')['label'].transform(
            'size') >= self.min_img_per_product]
        res_df = res.set_index('label')
        return res_df
    
    def __getitem__(self, index):
        place_id = self.places_ids[index]
        
        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]
        
        # sample K images (rows) from this place
        # we can either sort and take the most recent k images
        # or randomly sample them
        if self.random_sample_from_each_place:
            place = place.sample(n=self.img_per_product)
        else:  # always get the same most recent images
            place = place.sort_values(
                by=['year', 'month', 'lat'], ascending=False)
            place = place[: self.img_per_product]
            
        imgs = []
        for i, row in place.iterrows():
            img_name = self.get_img_name(row)
            img_path = self.base_path + 'Images/' + \
                row['city_id'] + '/' + img_name
            img = self.image_loader(img_path)

            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)
        # img:[3,320,320] ->imgs: [K,3,320,320], place_id: eg: 19, torch.tensor(place_id).repeat(self.img_per_product):[19,19,19,19]
        # 注意: 对比于图像分类，这里的__getitem__返回的是一个place，而不是一个图像
        # 它是一个Tesor of K images (K=self.img_per_product)
        # 这个Tensor的shape是[K, channels, height, width]，这需要在Dataloader中考虑到， 这将yield一个batch of shape [BS, K, channels, height, width]
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_product)

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def get_img_name(row):
        # given a row from the dataframe
        # return the corresponding image name

        city = row['city_id']
        
        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        pl_id = row.name % 10**5  #row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(7)
        
        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = city+'_'+pl_id+'_'+year+'_'+month+'_' + \
            northdeg+'_'+lat+'_'+lon+'_'+panoid+'.jpg'
        return name #获取图片的具体的名称
