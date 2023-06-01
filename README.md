# Image_Text_Multimodal

### 基于VSE模型的图文检索系统(2023-4-24 ~ 2023-5-28) 
> 模型架构基本参照论文[VSE++](https://github.com/fartashf/vsepp),我在其基础上做了复现以及另外的尝试。

仓库中并未放置images的图片信息，需要自行下载[斯坦福划分数据集](https://cs.stanford.edu/people/karpathy/deepimagesent/)。

后续可行的更新：(不急)
- 对于RNN中使用平均池化提升对句子抽取信息的能力方面，可以做点实验。试试对比非rnn_mean_pool模型在长短句上的表现效果以及rnn_mean_pool模型在长短句上的表现效果。
- 鉴于Bert对于整个句子信息提取上的欠佳表现，可以尝试使用SentenceBert与ResNet152集合试试，也可以看看T5试试






