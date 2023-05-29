from vocab import Vocabulary
from evaluation import evalrank
from param import args

evalrank('./model/model_best.pth.tar')
# 除了gpu号，其他的参数都会从存储的时候读取，所以只需额外指定gpu号(默认0)


# 返回指标： R@1 R@5 R@10 medi_rank mean_rank

"""
VGG19-margin0.15-model_best.pth.tar(--margin 0.15)
    R_sum : 276.0
    Average im2cap Recall: 47.9
    Image to text: 25.9 52.6 65.2 5.0 34.0
    Average cap2im Recall: 44.1
    Text to image: 21.5 49.1 61.8 6.0 33.0
"""

"""
VGG19-margin0.17-model_best.pth.tar(--margin 0.17)
    R_sum : 274.8
    Average im2cap Recall: 48.3
    Image to text: 24.6 54.3 65.9 5.0 31.0
    Average cap2im Recall: 43.3
    Text to image: 20.7 48.1 61.3 6.0 33.0
"""

"""
VGG19-margin0.2-model_best.pth.tar(--margin 0.2)
    R_sum : 274.7
    Average im2cap Recall: 48.4
    Image to text: 23.8 53.6 67.8 5.0 32.0
    Average cap2im Recall: 43.2
    Text to image: 20.1 48.0 61.4 6.0 32.0
"""

"""
VGG19-useWord2Vec-model_best.pth.tar(--use_word2vec)
    R_sum : 296.4
    Average im2cap Recall: 51.9
    Image to text: 28.8 57.4 69.4 4.0 27.0
    Average cap2im Recall: 46.9
    Text to image: 23.0 52.1 65.6 5.0 28.0
"""

"""
VGG19-margin0.3-model_best.pth.tar(--margin 0.3)
    R_sum : 262.3
    Average im2cap Recall: 45.8
    Image to text: 24.6 50.5 62.2 5.0 38.0
    Average cap2im Recall: 41.7
    Text to image: 18.8 46.0 60.2 7.0 35.0
"""

"""
VGG19-margin0.4-model_best.pth.tar(--margin 0.4)
    R_sum : 253.8
    Average im2cap Recall: 44.3
    Image to text: 22.2 49.7 61.1 6.0 40.0
    Average cap2im Recall: 40.3
    Text to image: 18.2 44.7 57.8 7.0 37.0
"""

"""
VGG19-rnnMeanPool-model_best.pth.tar(--margin 0.2 --rnn_mean_pool)
    R_sum : 286.2
    Average im2cap Recall: 50.7
    Image to text: 28.4 57.0 66.8 4.0 32.0
    Average cap2im Recall: 44.7
    Text to image: 21.0 49.6 63.3 6.0 32.0
"""

"""
VGG19-biGRU-rnnMeanPool-model_best.pth.tar(--rnn_mean_pool --bidirection_rnn)
    R_sum : 288.7
    Average im2cap Recall: 50.7
    Image to text: 27.7 56.1 68.2 4.0 31.0
    Average cap2im Recall: 45.6
    Text to image: 22.0 51.1 63.6 5.0 33.0
"""

"""
VGG19-useWord2Vec-model_best.pth.tar(--use_word2vec)
    R_sum : 296.4
    Average im2cap Recall: 51.9
    Image to text: 28.8 57.4 69.4 4.0 27.0
    Average cap2im Recall: 46.9
    Text to image: 23.0 52.1 65.6 5.0 28.0
"""

"""
resNet101_model_best.pth.tar(--use_InfoNCE_loss --rnn_mean_pool)
    R_sum : 341.0
    Average im2cap Recall: 61.1
    Image to text: 38.7 66.8 77.9 2.0 16.0
    Average cap2im Recall: 52.5
    Text to image: 29.3 58.1 70.3 4.0 24.0
"""

"""  
resNet152_model_best.pth.tar(--use_InfoNCE_loss --rnn_mean_pool --cnn_type 'ResNet152')
    R_sum : 341.1
    Average im2cap Recall: 60.4
    Image to text: 36.9 67.2 77.1 3.0 17.0
    Average cap2im Recall: 53.3
    Text to image: 29.5 59.5 71.0 3.0 23.0
"""

"""  
resNet101_biGRU_model_best.pth.tar(--use_InfoNCE_loss --rnn_mean_pool --bidirection_rnn --cnn_type 'ResNet101')
    R_sum : 347.5
    Average im2cap Recall: 62.0
    Image to text: 39.2 68.0 78.7 3.0 16.0
    Average cap2im Recall: 53.9
    Text to image: 30.2 59.7 71.7 3.0 21.0
"""

"""  
resNet152-biGRU-useWord2Vec-model_best.pth.tar(--use_InfoNCE_loss --rnn_mean_pool --bidirection_rnn --cnn_type 'ResNet152' --use_word2vec)
    R_sum : 363.5
    Average im2cap Recall: 64.2
    Image to text: 41.8 70.2 80.7 2.0 13.0
    Average cap2im Recall: 56.9
    Text to image: 32.5 63.6 74.7 3.0 17.0
"""

""" 
resNet152-biGRU-useWord2Vec-InfoNCE-finetune-model_best.pth.tar(--use_InfoNCE_loss --rnn_mean_pool --bidirection_rnn --cnn_type 'ResNet152' --use_word2vec --finetune)
    R_sum : 377.3
    Average im2cap Recall: 66.0
    Image to text: 43.5 72.4 82.1 2.0 14.0
    Average cap2im Recall: 59.8
    Text to image: 35.3 66.6 77.3 3.0 16.0
"""

"""  
resNet101-768-model_best.pth.tar(--use_InfoNCE_loss --rnn_mean_pool --cnn_type 'ResNet101' --embed_size 768)
    R_sum : 344.5
    Average im2cap Recall: 60.9
    Image to text: 39.4 66.1 77.1 2.0 19.0
    Average cap2im Recall: 54.0
    Text to image: 29.7 60.5 71.7 3.0 24.0
"""

"""  
resNet101-512-model_best.pth.tar(--use_InfoNCE_loss --rnn_mean_pool --cnn_type 'ResNet101' --embed_size 512)
    R_sum : 339.7
    Average im2cap Recall: 60.4
    Image to text: 38.9 65.6 76.7 3.0 16.0
    Average cap2im Recall: 52.8
    Text to image: 29.2 58.6 70.6 4.0 23.0
"""

"""  
resNet101-384-model_best.pth.tar(--use_InfoNCE_loss --rnn_mean_pool --cnn_type 'ResNet101' --embed_size 384)
    R_sum : 340.9
    Average im2cap Recall: 60.7
    Image to text: 40.0 65.8 76.2 2.0 17.0
    Average cap2im Recall: 53.0
    Text to image: 29.6 58.8 70.5 4.0 24.0
"""

"""  
resNet101-attention-useWord2Vec-model_best.pth.tar(--use_InfoNCE_loss --use_attention_for_text --cnn_type 'ResNet101' --use_word2vec)
    R_sum : 345.8
    Average im2cap Recall: 61.4
    Image to text: 38.3 68.0 77.9 2.0 17.0
    Average cap2im Recall: 53.9
    Text to image: 29.3 60.4 71.9 3.0 22.0
"""

"""  
resNet152-attention-useWord2Vec-model_best.pth.tar(--use_InfoNCE_loss --use_attention_for_text --cnn_type 'ResNet152' --use_word2vec)
    R_sum : 347.8
    Average im2cap Recall: 61.7
    Image to text: 39.1 66.9 79.1 2.0 16.0
    Average cap2im Recall: 54.2
    Text to image: 31.2 60.2 71.2 3.0 23.0
"""

"""  
resNet152-attention-finetune-model_best.pth.tar(--use_InfoNCE_loss --use_attention_for_text --cnn_type 'ResNet152' --use_word2vec --finetune)
    R_sum : 356.0
    Average im2cap Recall: 61.6
    Image to text: 39.1 67.6 78.1 2.0 15.0
    Average cap2im Recall: 57.1
    Text to image: 32.6 63.7 74.9 3.0 19.0
"""

"""resNet152-attention-rnnMeanPool-model_best.pth.tar(--use_InfoNCE_loss --use_attention_for_text --cnn_type 'ResNet152' --use_word2vec --rnn_mean_pool)
    R_sum : 335.6
    Average im2cap Recall: 58.9
    Image to text: 37.2 64.0 75.6 3.0 19.0
    Average cap2im Recall: 52.9
    Text to image: 29.0 59.2 70.6 4.0 21.0
"""

"""resNet152-attention-layers12-heads6-model_best.pth.tar(--use_InfoNCE_loss --use_attention_for_text --cnn_type 'ResNet152' --use_word2vec --num_heads 6 --num_layers 12)
    R_sum : 327.2
    Average im2cap Recall: 58.4
    Image to text: 36.9 63.5 74.8 3.0 18.0
    Average cap2im Recall: 50.7
    Text to image: 27.4 56.5 68.1 4.0 26.0
"""

"""resNet152-attention-rnnMeanPool-layers12-heads6-model_best.pth.tar(--use_InfoNCE_loss --use_attention_for_text --cnn_type 'ResNet152' --use_word2vec --num_heads 6 --num_layers 12 --rnn_mean_pool)
    R_sum : 331.7
    Average im2cap Recall: 59.4
    Image to text: 38.6 64.8 74.8 3.0 22.0
    Average cap2im Recall: 51.2
    Text to image: 27.5 57.0 69.0 4.0 25.0
"""

"""Bert2ViT-InfoNCE-model_best.pth.tar(--use_InfoNCE_loss --model_class 'ViT_and_BERT')
    R_sum : 351.1
    Average im2cap Recall: 63.6
    Image to text: 40.3 69.7 80.7 2.0 13.0
    Average cap2im Recall: 53.5
    Text to image: 29.0 59.6 71.7 4.0 20.0
"""

"""resNet152-bert-tmp.pth.tar(--use_InfoNCE_loss --model_class 'CNN_and_BERT')
    R_sum : 355.2
    Average im2cap Recall: 63.4
    Image to text: 41.2 69.9 79.2 2.0 13.0
    Average cap2im Recall: 55.0
    Text to image: 30.5 61.8 72.7 3.0 24.0
"""

"""resNet152-BiGRU-rnnMeanPool-useWord2Vec-maxViolation.pth.tar(--bidirection_rnn --rnn_mean_pool --cnn_type 'ResNet152' --use_word2vec --model_class 'CNN_and_GRU' --max_violation_in_middle)
    R_sum : 352.6
    Average im2cap Recall: 63.8
    Image to text: 42.3 69.0 80.1 2.0 12.0
    Average cap2im Recall: 53.7
    Text to image: 29.8 59.7 71.7 3.0 21.0
"""






