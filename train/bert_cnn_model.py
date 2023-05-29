import torch
import torch.nn as nn
from param import args
from transformers import BertModel
from bert_cnn_data_process import get_train_dev_loader , get_test_loader
import torch.nn.functional as F
from info_nce import InfoNCE
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
import torchvision.models as models
import numpy as np

if args.gpu.lower() == 'cpu':
    DEVICE = torch.device('cpu')
elif args.gpu in ['0' , '1' , '2' , '3']:
    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
else:
    raise ValueError('Invalid GPU ID')


class ImageEncoder(nn.Module):
    def __init__(self , embed_size , cnn_type , finetune=False):
        super(ImageEncoder , self).__init__()
        self.embed_size = embed_size
        if cnn_type == 'VGG19':
            self.cnn = models.vgg19(pretrained=True).to(DEVICE)
            # 在VGG19后面接一个映射层
            self.mapping = nn.Linear(
                self.cnn.classifier[6].out_features,
                embed_size,
                bias=True
            )
            # 设置VGG19的classifier是否参与训练
            for param in self.cnn.parameters():
                param.requires_grad = finetune

            # 分类层作为映射层参与训练
            for param in self.cnn.classifier.parameters():
                param.requires_grad = True

        elif 'ResNet' in cnn_type:
            if cnn_type == 'ResNet101':
                self.cnn = models.resnet101(pretrained=True).to(DEVICE)
            elif cnn_type == 'ResNet152':
                self.cnn = models.resnet152(pretrained=True).to(DEVICE)
            # 在ResNet后面接一个映射层
            self.mapping = nn.Linear(
                self.cnn.fc.out_features,
                embed_size,
                bias=True
            )
            # 设置ResNet的classifier是否参与训练
            for param in self.cnn.parameters():
                param.requires_grad = finetune
            # 分类层作为映射层参与训练
            for param in self.cnn.fc.parameters():
                param.requires_grad = True
        else:
            raise ValueError('Invalid model name') 
        if finetune: # 要预训练这个的话需要较大显存，我用的显卡(2080 11GB)上支持不了，需要在多个卡上并行
            """
                VGG19：共计需要约27GB显存
                ResNet152: 共计需要约34GB显存
            """
            self.cnn = nn.DataParallel(self.cnn)

        self.relu = nn.ReLU(inplace=True)

        # 使用Xavier初始化fc层
        r = np.sqrt(6.) / np.sqrt(self.mapping.in_features + self.mapping.out_features)
        self.mapping.weight.data.uniform_(-r, r)
        self.mapping.bias.data.fill_(0)

    def forward(self , images):
        features = self.cnn(images)
        features = self.relu(features)
        features = self.mapping(features)
        features = F.normalize(features , p=2 , dim=1)
        
        return features

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
        

    def forward(self , captions , lengths):
        captions = captions.input_ids.to(DEVICE)
        outputs = self.encoder(captions).last_hidden_state[: , 0 , :]
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


class BERT_CNN_VSE(object):
    def __init__(self , embed_size , finetune , margin , max_violation , grad_clip , use_InfoNCE_loss , cnn_type):
        self.grad_clip = grad_clip
        self.use_InfoNCE_loss = use_InfoNCE_loss
        self.image_encoder = ImageEncoder(embed_size , cnn_type , finetune).to(DEVICE)
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
    
    
