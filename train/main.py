import os
import torch
import time
import data_process
from vocab import Vocabulary
from model import VSE
from param import args
import logging
import tensorboard_logger as tb_logger
import numpy as np
from evaluation import im2cap , cap2im , AverageMeter , LogCollector , encode_data
import pickle
import shutil
from pprint import pprint


def main():
    logging.basicConfig(format='%(asctime)s %(message)s' , level=logging.INFO)
    tb_logger.configure(os.path.join(args.log_dir , f'GPU{args.gpu}') , flush_secs=5)
    # 加载词汇表
    vocab = pickle.load(open(os.path.join(args.vocab_path , 'flickr30k_vocab.pkl') , 'rb'))
    vocab_size = len(vocab)
    # 加载DataLoader
    train_loader , val_loader = data_process.get_train_dev_loader(
        vocab,
        args.batch_size,
        args.workers
    )
    # 构建模型
    model = VSE(
        args.embed_size,
        args.finetune,
        args.word_dim,
        args.num_layers,
        vocab_size,
        args.margin,
        args.max_violation,
        args.grad_clip,
        args.use_InfoNCE_loss,
        args.rnn_mean_pool,
        args.bidirection_rnn,
        args.cnn_type,
        args.use_attention_for_text,
        args.num_heads
    )

    # 训练模型
    best_rsum = 0
    for epoch in range(args.num_epochs):
        adjust_learning_rate(
            args.lr,
            model.optimizer,
            epoch
        )
        main_train(
            args.log_step,
            args.val_step,
            train_loader,
            val_loader,
            model,
            epoch
        )
        R_sum = validate(args.log_step , val_loader , model)

        is_best = R_sum > best_rsum
        best_rsum = max(R_sum , best_rsum)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'args': args,
                'whole_iters': model.whole_iters,
            }, 
            is_best, 
            prefix=args.log_dir + '/'
        )


def adjust_learning_rate(old_lr , optimizer , epoch):
    lr = old_lr * (0.5 ** (epoch // args.lr_decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main_train(log_step , val_step , train_loader , val_loader , model , epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    model.train_model()
    timer = time.time()
    for i , train_data in enumerate(train_loader):
        data_time.update(time.time() - timer)
        model.logger = train_logger
        images , caption_ids , lengths , _ = train_data
        model.train(images , caption_ids , lengths)
        batch_time.update(time.time() - timer)
        timer = time.time()

        # 打印日志
        if model.whole_iters % log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Loss: {e_log}\t'
                'Batch_time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data_time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch,
                    i,
                    len(train_loader),
                    e_log=str(model.logger.meters["Loss"]),
                    batch_time=batch_time,
                    data_time=data_time
                )
            )
        
        # 记录到tensorboard
        tb_logger.log_value('batch_time' , batch_time.val , model.whole_iters)
        tb_logger.log_value('data_time', data_time.val, model.whole_iters)
        model.logger.tensorboard_log(tb_logger , step=model.whole_iters)

        # 每经过val_step就验证
        if model.whole_iters % val_step == 0:
            validate(log_step , val_loader , model)

def validate(log_step , val_loader , model):
    img_embs , cap_embs = encode_data(
        model,
        val_loader,
        log_step,
        logging.info
    )
    #  以图搜文
    r_im2cap_1 , r_im2cap_5 , r_im2cap_10 , r_im2cap_medi , r_im2cap_mean = im2cap(
        img_embs,
        cap_embs
    )
    logging.info("Image to caption: %.1f, %.1f, %.1f, %.1f, %.1f" % (r_im2cap_1 , r_im2cap_5 , r_im2cap_10 , r_im2cap_medi , r_im2cap_mean))
    # 以文搜图
    r_cap2im_1 , r_cap2im_5 , r_cap2im_10 , r_cap2im_medi , r_cap2im_mean = cap2im(
        img_embs,
        cap_embs
    )
    logging.info("Caption to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r_cap2im_1 , r_cap2im_5 , r_cap2im_10 , r_cap2im_medi , r_cap2im_mean))

    # R_sum
    R_sum = r_im2cap_1 + r_im2cap_5 + r_im2cap_10 + r_cap2im_1 + r_cap2im_5 + r_cap2im_10

    # 记录到tensorboard上
    tb_logger.log_value('r_im2cap_1', r_im2cap_1, model.whole_iters)
    tb_logger.log_value('r_im2cap_5', r_im2cap_5, model.whole_iters)
    tb_logger.log_value('r_im2cap_10', r_im2cap_10, model.whole_iters)
    tb_logger.log_value('r_im2cap_medi', r_im2cap_medi, model.whole_iters)
    tb_logger.log_value('r_im2cap_mean', r_im2cap_mean, model.whole_iters)
    tb_logger.log_value('r_cap2im_1', r_cap2im_1, model.whole_iters)
    tb_logger.log_value('r_cap2im_5', r_cap2im_5, model.whole_iters)
    tb_logger.log_value('r_cap2im_10', r_cap2im_10, model.whole_iters)
    tb_logger.log_value('r_cap2im_medi', r_cap2im_medi, model.whole_iters)
    tb_logger.log_value('r_cap2im_mean', r_cap2im_mean, model.whole_iters)
    tb_logger.log_value('R_sum', R_sum, model.whole_iters)

    return R_sum

def save_checkpoint(state , is_best , file_name='checkpoint.pth.tar' , prefix='' ):
    torch.save(state , os.path.join(os.path.join(prefix , f'GPU{args.gpu}') , file_name))
    if is_best:
        shutil.copyfile(
            os.path.join(os.path.join(prefix , f'GPU{args.gpu}') , file_name), 
            os.path.join(os.path.join(prefix , f'GPU{args.gpu}') , 'model_best.pth.tar')
        )


if __name__ == '__main__':
    # 固定随机种子
    torch.manual_seed(16)
    torch.cuda.manual_seed(16)
    np.random.seed(16)
    # 开始训练
    print("-----------------------------------------------")
    pprint(args)
    print("-----------------------------------------------")

    main()

    print("-----------------------------------------------")
    pprint(args)
    print("-----------------------------------------------")