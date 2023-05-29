import torch
from torch.utils.data import Dataset , DataLoader
import os
from PIL import Image
import json
from param import args
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoFeatureExtractor


class FlickrDataset(Dataset):
    """
        对Flickr30k中读取到的数据进行预处理并打包为Dataset类型
    """
    def __init__(self , root , json_path , split):
        self.root = root
        self.dataset = json.load(open(json_path , 'r'))['images']
        self.ids = []
        for idx , dataItem in enumerate(self.dataset):
            if dataItem['split'] == split:
                self.ids += [(idx , capIdx) for capIdx in range(len(dataItem['sentences']))] # 一张图片形成5个图文对
                # [(i , x0) , (i , x1) , (i , x2) , (i , x3) , (i , x4)]
        
    def __getitem__(self , index):
        """
            根据索引，从dataset中抽取对应元素的张量数据，为collate_fn服务
        """
        pair_id = self.ids[index]
        img_id = pair_id[0]
        caption = self.dataset[img_id]['sentences'][pair_id[1]]['raw']

        img_path = self.dataset[img_id]['filename']
        image = Image.open(os.path.join(self.root , img_path)).convert('RGB')

        return image , caption , index
    
    def __len__(self):
        return len(self.ids)
    
def collate_fn(data):
    """
        为DataLoader服务，将从Dataset中抓取的数据进行打包
    """
    images , captions , ids = zip(*data)
    
    # 处理图像
    image_processor = AutoFeatureExtractor.from_pretrained("facebook/vit-msn-small")
    images = image_processor(images , return_tensors="pt")
    images = images['pixel_values']
    # 处理文本
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    lengths = [len(cap.split()) for cap in captions]
    captions = tokenizer(
        captions,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max(lengths)
    )
    
    return images , captions , lengths , list(ids)

def FlickrDataLoader(split , root , json_path , batch_size , shuffle , num_workers , collate_fn=collate_fn):
    """
        获取DataLoader
    """
    dataset = FlickrDataset(root , json_path , split)
    data_loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle, 
        num_workers = num_workers, 
        collate_fn = collate_fn
    )
    return data_loader  


def get_path(path='../data/'):
    imgdir = os.path.join(path , 'flickr30k-images')
    cap_path = os.path.join(path , 'dataset.json')

    return imgdir , cap_path


def get_train_dev_loader(batch_size , workers):
    imgdir , capdir = get_path()

    train_loader = FlickrDataLoader(
        split = 'train',
        root = imgdir,
        json_path = capdir,
        batch_size = batch_size,
        shuffle = True,
        num_workers = workers,
        collate_fn = collate_fn
    )
    
    val_loader = FlickrDataLoader(
        split = 'val',
        root = imgdir,
        json_path = capdir,
        batch_size = batch_size,
        shuffle = False,
        num_workers = workers,
        collate_fn = collate_fn
    )

    return train_loader , val_loader


def get_test_loader(batch_size , workers):
    imgdir , capdir = get_path()

    test_loader = FlickrDataLoader(
        split = 'test',
        root = imgdir,
        json_path = capdir,
        batch_size = batch_size,
        shuffle = False,
        num_workers = workers,
        collate_fn = collate_fn
    )

    return test_loader

if __name__ == "__main__":
    train_loader , val_loader = get_train_dev_loader(
        args.batch_size,
        args.workers
    )
    for idx , train_data in enumerate(train_loader):
        images , captions , lengths , _ = train_data
        if idx in [0,1,2,3]:
            print(images , images.shape) # tensor (batch_size , 3 , 224 , 224)
            print(captions) # dict: {input_ids: ... , attention_mask: ...}
            print(lengths) # list (batch_size)
        else:
            break



