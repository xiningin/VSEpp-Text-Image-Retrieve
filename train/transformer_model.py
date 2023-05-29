import torch
import torch.nn as nn
from param import args
# from transformers import AutoImageProcessor, ViTModel
from transformers import ViTMSNModel
from transformers import BertModel
from transformer_data_process import get_train_dev_loader , get_test_loader
import torch.nn.functional as F
from info_nce import InfoNCE
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
import numpy as np

if args.gpu.lower() == 'cpu':
    DEVICE = torch.device('cpu')
elif args.gpu in ['0' , '1' , '2' , '3']:
    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
else:
    raise ValueError('Invalid GPU ID')


class VITImageEncoder(nn.Module):
    def __init__(self , embed_size , finetune=False):
        super(VITImageEncoder , self).__init__()
        self.encoder = ViTMSNModel.from_pretrained("facebook/vit-msn-small")
        self.encoder = nn.DataParallel(self.encoder)
        self.relu = nn.ReLU(inplace=True)
        self.mapping = nn.Linear(384 , embed_size)

        # 使用Xavier初始化fc层
        r = np.sqrt(6.) / np.sqrt(self.mapping.in_features + self.mapping.out_features)
        self.mapping.weight.data.uniform_(-r, r)
        self.mapping.bias.data.fill_(0)

        # 多加几层映射
        # self.mapping = nn.Sequential(
        #     nn.Linear(384 , 384),
        #     nn.ReLU(),
        #     nn.Linear(384 , 384),
        #     nn.ReLU(),
        #     nn.Linear(384 , embed_size),
        # )


        for param in self.encoder.parameters():
            param.requires_grad = finetune

        
    def forward(self , images):
        outputs = self.encoder(images) 
        # print(outputs.last_hidden_state.shape)
        """
            (batch_size , 197 , 768 or 384) -> 
            197来源： 图片为224*224 ViT切块是每个块16*16，因此有14*14个块，加上一个类似CLS的类别头 14*14+1=197
        """
        outputs = outputs.last_hidden_state[: , 0 , :] # 取出CLS头代表整个图像
        outputs = self.relu(outputs)
        outputs = self.mapping(outputs)
        outputs = F.normalize(outputs , p=2 , dim=1)
        
        return outputs

class BertTextEncoder(nn.Module):
    def __init__(self , embed_size , finetune=False):
        super(BertTextEncoder , self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.encoder = nn.DataParallel(self.encoder)
        self.relu = nn.ReLU(inplace=True)
        self.mapping = nn.Linear(768 , embed_size)

        # 使用Xavier初始化fc层
        r = np.sqrt(6.) / np.sqrt(self.mapping.in_features + self.mapping.out_features)
        self.mapping.weight.data.uniform_(-r, r)
        self.mapping.bias.data.fill_(0)
        # self.mapping = nn.Sequential(
        #     nn.Linear(768 , 768),
        #     nn.ReLU(),
        #     nn.Linear(768 , 768),
        #     nn.ReLU(),
        #     nn.Linear(768 , embed_size),
        # )

        # Locked-image text Tuning论文的结论，锁住图像端，文本端向图像端对齐效果好
        # for param in self.encoder.parameters():
        #     param.requires_grad = finetune


    def forward(self , captions , lengths):
        captions = captions.input_ids.to(DEVICE)
        outputs = self.encoder(captions).last_hidden_state[: , 0 , :]
        # print(self.encoder(captions).last_hidden_state.shape) # torch.Size([128, 42, 768])
        outputs = self.relu(outputs)
        outputs = self.mapping(outputs)
        outputs = F.normalize(outputs , p=2 , dim=1)
        
        return outputs


def cosine_sim(im , cap): # 只有ContrastiveLoss类中使用
    """
        计算图片和文本的余弦相似度
        im : (batch_size , embed_dim)
        cap : (batch_size , embed_dim)
    """
    im_norm = im.norm(dim=1).unsqueeze(1).expand_as(im) + 1e-6
    cap_norm = cap.norm(dim=1).unsqueeze(1).expand_as(cap) + 1e-6
    sim_score = torch.mm(im/im_norm , (cap/cap_norm).t()) 
    return sim_score


class ContrastiveLoss(nn.Module):
    """
        计算对比损失
    """
    def __init__(self , margin , max_violation):
        super(ContrastiveLoss , self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self , im , cap):
        # 计算图片和文本的位置目标
        sim_score = cosine_sim(im , cap)
        positive = sim_score.diag().view(im.size(0), 1)
        posi_for_img = positive.expand_as(sim_score)
        posi_for_cap = positive.t().expand_as(sim_score)

        cost_cap = (self.margin + sim_score - posi_for_cap).clamp(min=0) # 以图搜文时
        cost_img = (self.margin + sim_score - posi_for_img).clamp(min=0) # 以文搜图时

        mask = torch.eye(sim_score.size(0)) > .5
        I = Variable(mask).to(DEVICE)

        cost_cap = cost_cap.masked_fill_(I , 0)
        cost_img = cost_img.masked_fill_(I , 0)

        if self.max_violation: # 试试先用一般训练，再后面使用max_violation(失败)
            cost_cap = cost_cap.max(1)[0] # 以图搜文时
            cost_img = cost_img.max(0)[0] # 以文搜图时
    
        return cost_cap.mean() + cost_img.mean()

class InfoNCE_contrastiveLoss(nn.Module):
    """
        使用InfoNCE对比损失
    """
    def __init__(self , temperature=0.1 , reduction='mean' , negative_mode='unpaired'):
        super(InfoNCE_contrastiveLoss , self).__init__()
        self.loss_calcer = InfoNCE(temperature , reduction , negative_mode)

    def forward(self , im , cap):
        all_loss = torch.zeros(im.shape[0] * 2) # (2*batch_size)
        # 以图搜文
        for index , im_item in enumerate(im):
            neg_cap = cap[torch.arange(cap.shape[0]) != index]
            all_loss[index] = self.loss_calcer(im_item.view(1,-1) , cap[index].view(1,-1) , neg_cap)
        # 以文搜图
        for index , cap_item in enumerate(cap):
            neg_im = im[torch.arange(im.shape[0]) != index]
            all_loss[cap.shape[0]+index] = self.loss_calcer(cap_item.view(1,-1) , im[index].view(1,-1) , neg_im)

        return all_loss.mean()


class transformer_VSE(object):
    def __init__(self , embed_size , finetune , margin , max_violation , grad_clip , use_InfoNCE_loss):
        self.grad_clip = grad_clip
        self.use_InfoNCE_loss = use_InfoNCE_loss
        self.image_encoder = VITImageEncoder(embed_size , finetune).to(DEVICE)
        self.text_encoder = BertTextEncoder(embed_size , finetune).to(DEVICE)
        self.margin = margin
        self.max_violation =max_violation
        self.temperature = nn.Parameter(torch.FloatTensor([args.temperature])) # 准备把这个系数加入训练
        self.params = list(self.image_encoder.parameters()) + list(self.text_encoder.parameters())
        self.params.append(self.temperature) # 加入温度系数
        self.optimizer = torch.optim.Adam(self.params , lr=args.lr)
        self.whole_iters = 0

    def state_dict(self):
        return [
            self.image_encoder.state_dict(),
            self.text_encoder.state_dict()
        ]

    def load_state_dict(self , state_dict):
        self.image_encoder.load_state_dict(state_dict[0])
        self.text_encoder.load_state_dict(state_dict[1])

    def train_model(self):
        self.image_encoder.train()
        self.text_encoder.train()

    def val_model(self):
        self.image_encoder.eval()
        self.text_encoder.eval()

    def forward(self , images , captions , lengths):
        """
            计算图片和文本的位置目标
        """
        if not self.use_InfoNCE_loss:
            self.contrastive_loss = ContrastiveLoss(self.margin , self.max_violation)
        else:
            self.contrastive_loss = InfoNCE_contrastiveLoss(
                self.temperature.cpu().item(),
                args.reduction,
            )

        images = images.to(DEVICE)
        # captions = captions.to(DEVICE) 此处captions是dict类型，在模型中再传入CUDA
        im_features = self.image_encoder(images)
        cap_features = self.text_encoder(captions , lengths)

        return im_features , cap_features
    
    def calc_loss(self , img_emb , cap_emb):
        """
            除了调用函数计算损失外，还得顺便记录logger
        """
        loss = self.contrastive_loss(img_emb , cap_emb)
        self.logger.update('Loss', loss.item() , img_emb.size(0))
        return loss

    def train(self , images , captions , lengths):
        self.train_model()

        self.whole_iters += 1
        self.logger.update('Iteration', self.whole_iters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        im_features , cap_features = self.forward(images , captions , lengths)
        self.optimizer.zero_grad()
        loss = self.calc_loss(im_features , cap_features)
        loss.backward()
        if self.grad_clip:
            clip_grad_norm_(self.params , self.grad_clip)
        self.optimizer.step()
    

if __name__ == "__main__":
    # model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    # print(model.ViTPooler)

    ######################### ViT部分 ####################
    # encoder = VITImageEncoder(
    #     args.embed_size,
    #     args.finetune
    # ).to(DEVICE)

    # train_loader , val_loader = get_train_dev_loader(
    #     args.batch_size,
    #     args.workers
    # )
    # for idx , train_data in enumerate(train_loader):
    #     images , captions , lengths , _ = train_data
    #     if idx == 1:
    #         with torch.no_grad():
    #             print(encoder(images.to(DEVICE)).shape)
    #         break

    ######################### BERT部分 ####################
    # encoder = BertTextEncoder(
    #     args.embed_size,
    #     args.finetune
    # ).to(DEVICE)

    # train_loader , val_loader = get_train_dev_loader(
    #     args.batch_size,
    #     args.workers
    # )
    # for idx , train_data in enumerate(train_loader):
    #     images , captions , lengths , _ = train_data
    #     if idx == 1:
    #         with torch.no_grad():
    #             print(encoder(captions , lengths).shape)
    #         break

    ################ 整体VSE部分 ####################

    # 构建模型
    model = transformer_VSE(
        args.embed_size,
        args.finetune, 
        args.margin,
        args.max_violation, 
        args.grad_clip,
        args.use_InfoNCE_loss       
    )

    train_loader , val_loader = get_train_dev_loader(
        args.batch_size,
        args.workers
    )
    for idx , train_data in enumerate(train_loader):
        images , captions , lengths , _ = train_data
        if idx == 1:
            with torch.no_grad():
                print(model.forward(images.to(DEVICE) , captions , lengths))
            break
    
