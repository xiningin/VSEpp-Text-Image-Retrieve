import os
import pickle
import torch
from param import args
from collections import OrderedDict
import numpy as np
import time
from data_process import get_test_loader
from model import VSE



if args.gpu.lower() == 'cpu':
    DEVICE = torch.device('cpu')
elif args.gpu in ['0' , '1' , '2' , '3']:
    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
else:
    raise ValueError('Invalid GPU ID')


class AverageMeter(object):
    """计算并且存储当前值和总计平均值(平滑)"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        if self.count == 0:
            return str(self.eval)
        return '%.4f (%.4f)' % (self.val, self.avg)
        
class LogCollector(object):
    """
        记录train和val的logging的对象
    """
    def __init__(self):
        self.meters = OrderedDict()
    def update(self, k, v, n=1):
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)
    def __str__(self):
        s = ''
        for i , (k , v) in enumerate(self.meters.items()):
            if i > 0:
                s += ' '
            s += f' {k} : {v}'
        return s
    def tensorboard_log(self , tb_logger , prefix='', step=None):
        for k , v in self.meters.items():
            tb_logger.log_value(prefix + k , v.val , step=step)

def im2cap(images , captions):
    image_nums = int(images.shape[0] / 5)
    ranks = np.zeros(image_nums)
    
    for index in range(image_nums):
        # 获取查询图片
        im = images[5 * index].reshape(1 , images.shape[1])
        # 计算相似度
        sim_score = np.dot(im , captions.T) # (1 , 5*image_nums) 

        inds = np.argsort(sim_score[0])[::-1]
        rank = 1e20
        for i in range(5 * index , 5 * index + 5 , 1): # 在5个匹配的文本中选取相似度最大的
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # 计算召回率指标
    r_im2cap_1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r_im2cap_5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r_im2cap_10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r_im2cap_medi = np.floor(np.median(ranks)) + 1
    r_im2cap_mean = np.floor(np.mean(ranks)) + 1

    return r_im2cap_1 , r_im2cap_5 , r_im2cap_10 , r_im2cap_medi , r_im2cap_mean

def cap2im(images , captions):
    image_nums = int(images.shape[0] / 5)
    ims = np.array([images[5 * i] for i in range(image_nums)]) # (image_nums , image_dim)
    ranks = np.zeros(images.shape[0])

    for index in range(image_nums):
        # 获取查询文本
        caps = captions[5*index : 5*index + 5]
        # 计算相似度
        sim_score = np.dot(caps , ims.T) # (5 , image_nums)
        inds = np.zeros(sim_score.shape)
        
        for i in range(len(inds)):
            inds[i] = np.argsort(sim_score[i])[::-1]
            ranks[5*index + i] = np.where(inds[i] == index)[0][0]
    
    # 计算召回率指标
    r_cap2im_1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r_cap2im_5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r_cap2im_10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r_cap2im_medi = np.floor(np.median(ranks)) + 1
    r_cap2im_mean = np.floor(np.mean(ranks)) + 1

    return r_cap2im_1 , r_cap2im_5 , r_cap2im_10 , r_cap2im_medi , r_cap2im_mean    

def encode_data(model , data_loader , log_step=10 , logging=print):
    batch_time = AverageMeter()
    val_logger = LogCollector()
    model.val_model()

    timer = time.time()
    img_embs = None
    cap_embs = None
    isInit = False

    for i , (images , captions , lengths , ids) in enumerate(data_loader):
        model.logger = val_logger

        with torch.no_grad():
            img_emb , cap_emb = model.forward(images , captions , lengths)

            # 初始化
            if not isInit:
                img_embs = np.zeros((len(data_loader.dataset) , img_emb.shape[1]))
                cap_embs = np.zeros((len(data_loader.dataset) , cap_emb.shape[1]))
                isInit = True

            img_embs[ids] = img_emb.data.cpu().numpy().copy()
            cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

            model.calc_loss(img_emb , cap_emb)
            # del images, captions # 有用嘛说实话 ？？？

        batch_time.update(time.time() - timer)

        if i % log_step == 0:
            logging(
                'Test: [{0}/{1}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                    i , len(data_loader) ,
                    e_log = model.logger ,
                    batch_time = batch_time
                )
            )

            timer = time.time()

    return img_embs , cap_embs

def evalrank(model_path):
    checkpoint = torch.load(model_path)
    old_args = checkpoint['args']

    with open(os.path.join(old_args.vocab_path , 'flickr30k_vocab.pkl') , 'rb') as f:
        vocab = pickle.load(f)
    old_args.vocab_size = len(vocab)
    model = VSE(
        old_args.embed_size,
        old_args.finetune,
        old_args.word_dim,
        old_args.num_layers,
        old_args.vocab_size,
        old_args.margin,
        old_args.max_violation,
        old_args.grad_clip
    )
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(
        vocab,
        old_args.batch_size,
        old_args.workers
    )

    print('Computing result....')
    img_embs , cap_embs = encode_data(model , data_loader)
    print(f'Images: {img_embs.shape[0] / 5} , Captions: {cap_embs.shape[0]}')

    r_im2cap = im2cap(img_embs , cap_embs)
    r_cap2im = cap2im(img_embs , cap_embs)

    ave_r_im2cap = (r_im2cap[0] + r_im2cap[1] + r_im2cap[2]) / 3
    ave_r_cap2im = (r_cap2im[0] + r_cap2im[1] + r_cap2im[2]) / 3
    R_sum = r_im2cap[0] + r_im2cap[1] + r_im2cap[2] + r_cap2im[0] + r_cap2im[1] + r_cap2im[2]

    print("----------------------------------------")
    print("R_sum : %.1f" % R_sum)
    print("Average im2cap Recall: %.1f" % ave_r_im2cap)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r_im2cap)
    print("Average cap2im Recall: %.1f" % ave_r_cap2im)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % r_cap2im)
        

    

    
    


