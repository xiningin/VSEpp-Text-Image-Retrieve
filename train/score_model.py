from vocab import Vocabulary
from evaluation import evalrank
from param import args

evalrank('./model/model_best.pth.tar')


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
VGG19-margin0.2-model_best.pth.tar(--margin 0.2)
    R_sum : 274.7
    Average im2cap Recall: 48.4
    Image to text: 23.8 53.6 67.8 5.0 32.0
    Average cap2im Recall: 43.2
    Text to image: 20.1 48.0 61.4 6.0 32.0
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
    R_sum : 276.2
    Average im2cap Recall: 48.4
    Image to text: 25.4 53.4 66.5 5.0 35.0
    Average cap2im Recall: 43.6
    Text to image: 21.0 48.4 61.5 6.0 34.0
"""

"""
resNet101-768-model_best.pth.tar(--use_InfoNCE_loss --rnn_mean_pool --cnn_type 'ResNet101' --embed_size 768)
    R_sum : 284.9
    Average im2cap Recall: 50.6
    Image to text: 27.7 56.3 67.7 4.0 34.0
    Average cap2im Recall: 44.4
    Text to image: 21.3 49.1 62.7 6.0 35.0
"""

"""
resNet101-512-model_best.pth.tar(--use_InfoNCE_loss --rnn_mean_pool --cnn_type 'ResNet101' --embed_size 768)
    R_sum : 282.6
    Average im2cap Recall: 50.1
    Image to text: 28.3 55.7 66.3 4.0 33.0
    Average cap2im Recall: 44.1
    Text to image: 20.9 49.1 62.2 6.0 33.0
"""

"""
resNet101-384-model_best.pth.tar(--use_InfoNCE_loss --rnn_mean_pool --cnn_type 'ResNet101' --embed_size 768)
    R_sum : 281.1
    Average im2cap Recall: 49.8
    Image to text: 25.8 55.8 67.9 4.0 34.0
    Average cap2im Recall: 43.9
    Text to image: 20.5 49.3 61.9 6.0 36.0
"""

