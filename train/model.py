import torch 
import torch.nn as nn
import torchvision.models as models
from param import args
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from info_nce import InfoNCE
import gensim
import os


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
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1]
            )
            """out_features
            VGG19的classifier
                (classifier): Sequential(
                    (0): Linear(in_features=25088, out_features=4096, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.5, inplace=False)
                    (3): Linear(in_features=4096, out_features=4096, bias=True)
                    (4): ReLU(inplace=True)
                    (5): Dropout(p=0.5, inplace=False)
                    (6): Linear(in_features=4096, out_features=1000, bias=True)
                )
            """
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
                """
                ResNet101的classifier
                    (fc): Linear(in_features=2048, out_features=1000, bias=True)
                """
            elif cnn_type == 'ResNet152':
                self.cnn = models.resnet152(pretrained=True).to(DEVICE)

            # 在ResNet后面接一个映射层
            self.mapping = nn.Linear(
                self.cnn.fc.in_features,
                embed_size,
                bias=True
            )
            self.cnn.fc = nn.Sequential()

            # 设置ResNet是否参与训练
            for param in self.cnn.parameters():
                param.requires_grad = finetune

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
    
class TextEncoder(nn.Module):
    def __init__(self , vocab , word_dim , embed_size , num_layers , rnn_mean_pool , bidirection_rnn , use_word2vec):
        super(TextEncoder , self).__init__()
        self.embed_size = embed_size
        self.embed = nn.Embedding(len(vocab) , word_dim)
        weights = torch.zeros(len(vocab), word_dim)
        if use_word2vec:
            if os.path.exists('../word2vec/google-news-300.pt'):
                weights = torch.load('../word2vec/google-news-300.pt')
            else: # 重新从预训练词向量赋值embedding需要接近一分钟
                weights.uniform_(-0.1, 0.1)
                print("Loading word vectors from word2vec-google-news-300!!!")
                word_vectors = gensim.models.KeyedVectors.load_word2vec_format('../word2vec/GoogleNews-vectors-negative300.bin.gz', binary=True)
                for word , idx in vocab.word2idx.items():
                    if word == '<start>' or word == '<end>':
                        weights[idx] = torch.FloatTensor(word_vectors['</s>'])
                        continue
                    if word not in word_vectors.key_to_index:
                        continue
                    weights[idx] = torch.FloatTensor(word_vectors[word])
            print("Finished loading!!!")
        # torch.save(weights , '../word2vec/google-news-300.pt')
        self.embed.weight = nn.Parameter(weights)
            
        if bidirection_rnn:        
            self.rnn = nn.GRU(word_dim , embed_size , num_layers , batch_first=True , bidirectional=True)
            self.linear = nn.Linear(2 * embed_size , embed_size)
        else:
            self.rnn = nn.GRU(word_dim , embed_size , num_layers , batch_first=True)
            self.linear = nn.Linear(embed_size , embed_size)            
        self.rnn_mean_pool = rnn_mean_pool

    def forward(self , text , lengths):
        embed = self.embed(text)
        packed = pack_padded_sequence(embed , lengths , batch_first=True)
        output , _ = self.rnn(packed)
        output , _ = pad_packed_sequence(output , batch_first=True) 

        if self.rnn_mean_pool:
            I = torch.LongTensor(lengths).view(-1,1)
            output = torch.sum(output , dim=1)
            output = torch.div(output , I.expand_as(output).to(DEVICE))
        else:
            pass
        
        output = self.linear(output)
        output = F.normalize(output , p=2 , dim=1)

        return output


class Attention_TextEncoder(nn.Module):
    def __init__(self , vocab , word_dim , embed_size , num_heads , num_layers , use_word2vec , rnn_mean_pool):
        super(Attention_TextEncoder , self).__init__()
        # Embedding层
        self.embed = nn.Embedding(len(vocab) , word_dim)
        weights = torch.zeros(len(vocab), word_dim)
        if use_word2vec:
            if os.path.exists('../word2vec/google-news-300.pt'):
                weights = torch.load('../word2vec/google-news-300.pt')
            else: # 重新从预训练词向量赋值embedding需要接近一分钟
                weights.uniform_(-0.1, 0.1)
                print("Loading word vectors from word2vec-google-news-300!!!")
                word_vectors = gensim.models.KeyedVectors.load_word2vec_format('../word2vec/GoogleNews-vectors-negative300.bin.gz', binary=True)
                for word , idx in vocab.word2idx.items():
                    if word == '<start>' or word == '<end>':
                        weights[idx] = torch.FloatTensor(word_vectors['</s>'])
                        continue
                    if word not in word_vectors.key_to_index:
                        continue
                    weights[idx] = torch.FloatTensor(word_vectors[word])
            print("Finished loading!!!")
        self.embed.weight = nn.Parameter(weights)
        atte_layer = nn.TransformerEncoderLayer(d_model=word_dim , nhead=num_heads) 
        self.encoder = nn.TransformerEncoder(atte_layer , num_layers=num_layers) # 输入数据的形状为(seq_length, batch_size, d_model)

        # 映射对齐层
        self.linear = nn.Linear(word_dim , embed_size)

        self.rnn_mean_pool = rnn_mean_pool
    
    def forward(self , x , lengths):
        x = self.embed(x)
        x = self.encoder(x.transpose(0 , 1)) # (seq_length, batch_size, d_model) -> torch.Size([31, 128, 300])
        x = x.transpose(0 , 1) # -> torch.Size([128, 31, 300])
        if self.rnn_mean_pool:
            I = torch.LongTensor(lengths).view(-1 , 1)
            # 创建掩码张量
            mask = (torch.arange(x.shape[1])[None , :] < I[: , None]).float().squeeze(1).to(DEVICE)
            output_masked = x * mask.unsqueeze(-1)
            x = torch.sum(output_masked , dim=1)
            x = torch.div(x , I.expand_as(x).to(DEVICE)) # 与之前的GRU一样所有有效加和
            x = self.linear(x) # 映射到对其层
        else:
            x = self.linear(x[: , 0 , :]) # 取第一个代表整个句子的含义

        x = F.normalize(x , p=2 , dim=1)
        return x

        
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
            # print(cost_cap , cost_cap.shape)
            # print(cost_img , cost_img.shape)

        # if cost_cap.mean() + cost_img.mean() == torch.FloatTensor([0.4000]).to(DEVICE):
        #     # 很容易最大的那一个直接与负样本拉开0.2了， 把sim_score和posi_for_cap学成0
        #     print(sim_score)
        #     print(posi_for_cap)
    
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


# 用CLIP中的伪代码重写了一个损失函数试试
# class InfoNCE_contrastiveLoss(nn.Module):
#     """
#         使用InfoNCE对比损失
#     """
#     def __init__(self , temperature=0.1 , reduction='mean'):
#         super(InfoNCE_contrastiveLoss , self).__init__()
#         self.temperature = temperature
#         self.reduction = reduction

#     def forward(self , im , cap):
#         # 模型forward中最后都是normalize
#         # Im_e = F.normalize(im , p=2 , dim=1)
#         # Cap_e = F.normalize(cap , p=2 , dim=1)
#         logits = torch.matmul(im , cap.T) * torch.exp(torch.tensor(self.temperature))
#         labels = torch.arange(im.shape[0]).to(DEVICE)
#         loss_im = F.cross_entropy(logits , labels , reduction=self.reduction)
#         loss_cap = F.cross_entropy(logits.T , labels , reduction=self.reduction)

#         return (loss_im + loss_cap) / 2
    
class VSE(object):
    def __init__(
        self , embed_size , finetune ,  word_dim , num_layers , vocab , 
        margin , max_violation , grad_clip , use_InfoNCE_loss , rnn_mean_pool , 
        bidirection_rnn , use_word2vec, cnn_type , use_attention_for_text , num_heads
    ):
        self.margin = margin
        self.max_violation = max_violation
        self.grad_clip = grad_clip
        self.use_InfoNCE_loss = use_InfoNCE_loss
        self.image_encoder = ImageEncoder(embed_size , cnn_type , finetune).to(DEVICE)
        if not use_attention_for_text:
            self.text_encoder = TextEncoder(vocab , word_dim , embed_size , num_layers , rnn_mean_pool , bidirection_rnn , use_word2vec).to(DEVICE)
        else:
            self.text_encoder = Attention_TextEncoder(vocab , word_dim , embed_size , num_heads , num_layers , use_word2vec , rnn_mean_pool).to(DEVICE)
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
        captions = captions.to(DEVICE)
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

    
