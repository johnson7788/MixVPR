import pytorch_lightning as pl
import json
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from dataloaders.ComesticDataset import ComesticDataset
from . import CosDataset

from prettytable import PrettyTable

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 
                'std': [0.5, 0.5, 0.5]}

# 读取/home/wac/johnson/johnson/MixVPR/datasets/cos_val/test.json中的品牌
with open('datasets/cos_val/test.json', 'r') as f:
    brands_data = json.load(f)
BRANDS = list(brands_data.keys())

class ComesticDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=32,
                 img_per_product=4,
                 min_img_per_product=4,
                 shuffle_all=False,
                 image_size=(480, 640),
                 num_workers=0,
                 show_data_stats=True,
                 brands=BRANDS,
                 mean_std=IMAGENET_MEAN_STD,
                 batch_sampler=None,
                 random_sample_from_each_product=True,
                 val_set_names=['cos_val']
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.img_per_product = img_per_product
        self.min_img_per_product = min_img_per_product
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.show_data_stats = show_data_stats
        self.brands = brands
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.random_sample_from_each_product = random_sample_from_each_product
        self.val_set_names = val_set_names
        self.save_hyperparameters() # save hyperparameter with Pytorch Lightening

        self.train_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)])

        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': self.shuffle_all}

        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers//2,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}

    def setup(self, stage):
        if stage == 'fit':
            # load train dataloader with reload routine
            self.reload()

            # load validation sets (pitts_val, msls_val, ...etc)
            self.val_datasets = []
            for valid_set_name in self.val_set_names:
                if valid_set_name.lower() == 'cos_val':
                    self.val_datasets.append(CosDataset.COS(
                        input_transform=self.valid_transform))
                else:
                    print(
                        f'Validation set {valid_set_name} does not exist or has not been implemented yet')
                    raise NotImplementedError
            if self.show_data_stats:
                self.print_stats()

    def reload(self):
        self.train_dataset = ComesticDataset(
            brands=self.brands,
            img_per_product=self.img_per_product,
            min_img_per_product=self.min_img_per_product,
            random_sample_from_each_product=self.random_sample_from_each_product,
            transform=self.train_transform)

    def train_dataloader(self):
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(DataLoader(
                dataset=val_dataset, **self.valid_loader_config))
        return val_dataloaders

    def print_stats(self):
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["# of brands", f"{len(BRANDS)}"])
        table.add_row(["# of products", f'{self.train_dataset.__len__()}'])
        table.add_row(["# of images", f'{self.train_dataset.total_nb_images}'])
        print(table.get_string(title="Training Dataset"))
        print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        for i, val_set_name in enumerate(self.val_set_names):
            table.add_row([f"Validation set {i+1}", f"{val_set_name}"])
        # table.add_row(["# of places", f'{self.train_dataset.__len__()}'])
        print(table.get_string(title="Validation Datasets"))
        print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(
            ["Batch size (PxK)", f"{self.batch_size}x{self.img_per_product}"])
        table.add_row(
            ["# of iterations", f"{self.train_dataset.__len__()//self.batch_size}"])
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))
