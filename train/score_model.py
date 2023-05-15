from vocab import Vocabulary
from evaluation import evalrank
from param import args

evalrank('./model/resNet101_biGRU_model_best.pth.tar')



"""
resNet101_model_best.pth.tar(--use_InfoNCE_loss --rnn_mean_pool)
    R_sum : 286.9
    Average im2cap Recall: 50.6
    Image to text: 28.3 55.9 67.5 4.0 31.0
    Average cap2im Recall: 45.1
    Text to image: 21.5 50.0 63.7 6.0 31.0
"""

"""
resNet152_model_best.pth.tar(--use_InfoNCE_loss --rnn_mean_pool --cnn_type 'ResNet152')
    R_sum : 289.1
    Average im2cap Recall: 50.9
    Image to text: 28.7 55.9 68.1 4.0 34.0
    Average cap2im Recall: 45.5
    Text to image: 21.7 51.1 63.6 5.0 31.0
"""

"""
resNet101_biGRU_model_best.pth.tar(--use_InfoNCE_loss --rnn_mean_pool --bidirection_rnn --cnn_type 'ResNet101')
    R_sum : 276.2
    Average im2cap Recall: 48.4
    Image to text: 25.4 53.4 66.5 5.0 35.0
    Average cap2im Recall: 43.6
    Text to image: 21.0 48.4 61.5 6.0 34.0
"""