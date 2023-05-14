import torch
from torch.utils.data import Dataset , DataLoader
import os
import torchvision.transforms as transforms
import nltk
from PIL import Image
import json


class FlickrDataset(Dataset):
    """
        对Flickr30k中读取到的数据进行预处理并打包为Dataset类型
    """
    def __init__(self , root , json_path , split , vocab , transform=None):
        self.root = root
        self.vocab = vocab
        self.transform = transform
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

        # 预处理图片信息
        if self.transform is not None:
            image = self.transform(image)
        # 预处理文本信息
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower()
        )
        caption_ids = []
        caption_ids.append(self.vocab('<start>'))
        caption_ids.extend([self.vocab(token) for token in tokens])
        caption_ids.append(self.vocab('<end>'))
        caption_ids_tensor = torch.Tensor(caption_ids)

        return image , caption_ids_tensor , index
    
    def __len__(self):
        return len(self.ids)
    
def collate_fn(data):
    """
        为DataLoader服务，将从Dataset中抓取的数据进行打包
    """
    data.sort(key = lambda x : len(x[1]), reverse = True) # 按照文本长度降序排序
    images , captions , ids = zip(*data)
    
    images = torch.stack(images , dim = 0)

    lengths = [len(cap) for cap in captions]
    caption_ids = torch.zeros(len(captions) , max(lengths) , dtype = torch.long)
    for index , cap in enumerate(captions):
        end = lengths[index]
        caption_ids[index , :end] = cap[:end]
    
    return images , caption_ids , lengths , list(ids)

def FlickrDataLoader(split , root , json_path , vocab , transform , batch_size , shuffle , num_workers , collate_fn=collate_fn):
    """
        获取DataLoader
    """
    dataset = FlickrDataset(root , json_path , split , vocab , transform)
    data_loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle, 
        num_workers = num_workers, 
        collate_fn = collate_fn
    )
    return data_loader  


def get_transform(split):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_list = []
    
    if split == 'train':
        transform_list.append(transforms.RandomResizedCrop(224))
        transform_list.append(transforms.RandomHorizontalFlip())
    elif split == 'val' or split == 'test':
        transform_list.append(transforms.Resize(256))
        transform_list.append(transforms.CenterCrop(224))

    transform_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(transform_list + transform_end)

    return transform


def get_path(path='../data/'):
    imgdir = os.path.join(path , 'flickr30k-images')
    cap_path = os.path.join(path , 'dataset.json')

    return imgdir , cap_path


def get_train_dev_loader(vocab , batch_size , workers):
    imgdir , capdir = get_path()

    transform = get_transform('train')
    train_loader = FlickrDataLoader(
        split = 'train',
        root = imgdir,
        json_path = capdir,
        vocab = vocab,
        transform = transform,
        batch_size = batch_size,
        shuffle = True,
        num_workers = workers,
        collate_fn = collate_fn
    )
    
    transform = get_transform('val')
    val_loader = FlickrDataLoader(
        split = 'val',
        root = imgdir,
        json_path = capdir,
        vocab = vocab,
        transform = transform,
        batch_size = batch_size,
        shuffle = False,
        num_workers = workers,
        collate_fn = collate_fn
    )

    return train_loader , val_loader


def get_test_loader(vocab , batch_size , workers):
    imgdir , capdir = get_path()

    transform = get_transform('test')
    test_loader = FlickrDataLoader(
        split = 'test',
        root = imgdir,
        json_path = capdir,
        vocab = vocab,
        transform = transform,
        batch_size = batch_size,
        shuffle = False,
        num_workers = workers,
        collate_fn = collate_fn
    )

    return test_loader


