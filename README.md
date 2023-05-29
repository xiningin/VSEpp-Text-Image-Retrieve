# Image_Text_Multimodal

- 基于VSE模型的图文检索系统(2023-4-24 ~ 2023-5-28) 
> 模型架构基本参照论文[VSE++](https://github.com/fartashf/vsepp),我在其基础上做了复现以及另外的尝试。

代码整体架构以及需要新建的文件夹为：
train:
--model
--nohup_logger
--tensorboard_logger
&emsp;   --GPU0
&emsp;   --GPU1
&emsp;   --GPU2
&emsp;   --GPU3
--vocab
--word2vec


data:
dataset.json
--images

仓库中并未放置images的图片信息，需要自行下载[斯坦福划分数据集](https://cs.stanford.edu/people/karpathy/deepimagesent/)。






