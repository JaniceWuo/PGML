import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import logging
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.reset_default_graph()
from tensorflow.python import pywrap_tensorflow
from copy import deepcopy
import argparse
from time import time, sleep
import json
import numpy as np
from data_loader import Batcher,Batcher_
import utils
from models.basemodel import BaseModel
from models.lm import LMModel
from models.emb_min import DiverEmbMin
from beamsearch import BeamsearchDecoder
from random import shuffle
import math


parser = argparse.ArgumentParser()

# path for data and models storage
parser.add_argument("--parsed_data_path", type=str, default="data/trainable/split/parsed_perspectrum_data.json")
parser.add_argument("--processed_data_path", type=str, default="data/trainable/split/processed_perspectrum_data.json")
parser.add_argument("--split_data_path", type=str, default="data/trainable/split/{}_processed.json")
parser.add_argument('--wikitext_raw_path', type=str, default='data/wikitext/wikitext-103/wiki.{}.tokens')
parser.add_argument("--data_path", type=str, default="data/trainable/split/train_processed.bin", help="Path to binarized train/valid/test data.")
parser.add_argument("--vocab_path", type=str, default="data/vocab.txt", help="Path to vocabulary.")
parser.add_argument("--embed_path", type=str, default="data/emb/glove.6B.300d.txt", help="Path to word embedding.")
parser.add_argument("--custom_embed_path", type=str, default="data/emb/my_words.txt")
parser.add_argument("--model_path", type=str, default="data/log/{}", help="Path to store the models checkpoints.")
parser.add_argument("--exp_name", type=str, default="scratch", help="Experiment name under model_path.")
parser.add_argument("--parser_path", type=str, default="./stanford-corenlp-full-2018-10-05")
parser.add_argument("--pretrain_ckpt_path", type=str, default='./data/log/lm/scratch/train/')
parser.add_argument("--true_pretrain_ckpt_path", type=str, default='./pretrain/model/') 
parser.add_argument("--init_ckpt_path", type=str, default='./save/')
parser.add_argument('--gpu_nums', type=str, default='0,1', help='gpu id to use')  

# models setups
parser.add_argument("--model", type=str, choices=["vanilla", 'lm', 'embmin', 'mmi_bidi'], default="vanilla", help="Different types of models, choose from vanilla, sep_dec, and shd_dec.")
parser.add_argument("--mode", type=str, choices=["train", "lm_train", "decode", 'eval'], help="Whether to run train, eval, or decode", default="train")
parser.add_argument("--min_cnt", type=int, help="word minimum count", default=1)
parser.add_argument("--use_pretrain", type=str, choices=['True', 'False'], default='True')

parser.add_argument('--beam_size', type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_enc_len", type=int, default=50)
parser.add_argument("--max_dec_len", type=int, default=50)
parser.add_argument("--max_lm_len", type=int, default=150)
parser.add_argument("--learning_rate", type=float, default=5e-4) 
parser.add_argument("--meta_learning_rate", type=float, default=5e-3)
parser.add_argument("--rand_unif_init_size", type=float, default=1e-3)
parser.add_argument("--trunc_norm_init_std", type=float, default=1e-3)
parser.add_argument("--max_grad_norm", type=float, default=3)
parser.add_argument("--vocab_size", type=int, default=50000)
parser.add_argument("--embed_dim", type=int, default=300)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--decoder_hidden_dim", type=int, default=384)
parser.add_argument("--keep_rate", type=float, default=0.8)
parser.add_argument("--max_epoch", type=int, default=25) 

parser.add_argument("--matrix_num", type=int, default=10)
parser.add_argument("--matrix_dim", type=int, default=128)
parser.add_argument('--gumbel_temp', type=float, default=0.67)
parser.add_argument('--use_aux_task', type=str, default='True')
parser.add_argument('--kl_coeff', type=float, default=0.5)
parser.add_argument('--aux_coeff', type=float, default=1.0)
parser.add_argument('--reg_coeff', type=float, default=0.05)

parser.add_argument('--multihead_num', type=int, default=5)

parser.add_argument('--mmi_bsize', type=int, default=100)
parser.add_argument('--mechanism_dim', type=int, default=128) 
parser.add_argument('--mmi_lambda', type=float, default=0.5)
parser.add_argument('--mmi_gamma', type=float, default=1.0)

parser.add_argument('--emb_min_coeff', type=float, default=0.5)
parser.add_argument('--word_min_coeff', type=float, default=0.6)
parser.add_argument("--meta_batch_size", type=int, default=16)
parser.add_argument("--wordnum", type=int, default=1000)
args = parser.parse_args()
epoch_batch_size = 15

# tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
# print ('TPU address is', tpu_address)

def do_learning_fix_step(model, train_iter, val_iter, sess):
    
    val_p = []
    val_p_list = []
    val_loss = 0
    
    res_train = model.run_step(train_iter, sess, is_train=True, freeze_layer=model.hps.use_pretrain)
    
    res_val,val_batch_loss = model.run_step(val_iter, sess, is_train=False, freeze_layer=model.hps.use_pretrain,val=True) 
    # print("val_batch_loss",val_batch_loss)
    
    return res_val,val_batch_loss

#reset model
def reset_model():
    modeldicts = utils.get_pretrain_weights(args.init_ckpt_path)
    assign_ops, uninitialized_varlist = utils.assign_pretrain_weights(modeldicts)
    return assign_ops,uninitialized_varlist
    
def deal_gradient(grad):
    for i in range(len(grad))[::-1]:
        if grad[i][0] is None:
            del grad[i]
    return grad

def train(model, vocab, pretrain_vardicts=None): 
    
    train_data_loader = Batcher_(vocab, model.hps.data_path, args) 
    valid_data_loader = Batcher_(vocab, model.hps.data_path.replace('train_', 'dev_'), args)
    all_id,claim_id,claim_pers = train_data_loader.get_claim_id()
    all_val_id,claim_val_id,claim_val_pers = valid_data_loader.get_claim_id()

    train_logdir, dev_logdir = os.path.join(args.model_path, 'logdir/train'), os.path.join(args.model_path, 'logdir/dev')
    train_savedir = os.path.join(args.model_path, 'train/')
    print("[*] Train save directory is: {}".format(train_savedir))
    if not os.path.exists(train_logdir): os.makedirs(train_logdir)
    if not os.path.exists(dev_logdir): os.makedirs(dev_logdir)
    if not os.path.exists(train_savedir): os.makedirs(train_savedir)
    # print(all_id)

    los = model.get_loss()
    
    optim = tf.train.MomentumOptimizer(model.hps.meta_learning_rate,0.9)
    grads_vars = optim.compute_gradients(los,tf.trainable_variables(),aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)  
    grads_vars = deal_gradient(grads_vars)
    grads_cache = [tf.Variable(np.zeros(t[0].shape.as_list(), np.float32), trainable=False) for t in grads_vars[1:]]
    clear_grads_cache_op = tf.group([gc.assign(tf.zeros_like(gc)) for gc in grads_cache])
    accumulate_grad_op = tf.group([gc.assign_add(gv[0]) for gc, gv in zip(grads_cache, grads_vars[1:])])
    new_grads_vars = [(g, gv[1]) for g, gv in zip(grads_cache, grads_vars[1:])]
    apply_grad_op = optim.apply_gradients(new_grads_vars)
    print("ready done!")      


    with tf.Session(config=utils.gpu_config()) as sess:     
       
        if model.hps.use_pretrain:
            assign_ops, uninitialized_varlist = utils.assign_pretrain_weights(pretrain_vardicts)
            sess.run(assign_ops)
            sess.run(tf.initialize_variables(uninitialized_varlist))
        else:
            sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        model.saver.save(sess,'./save/model')
        with tf.device('/cpu:0'):
            saver_ = tf.train.import_meta_graph('./save/model.meta')
        
        for meta_iteration in range(model.hps.max_epoch):     
            train_loss_before = []
            train_loss_meta = []
            epoch_val_loss = 0
            for epoch_bs in range(epoch_batch_size):
                batch_loss=0
                val_all_loss = tf.zeros((), dtype=tf.float32)
                val_all_dia = []
                batch_grad_list = []
                sess.run(clear_grads_cache_op)
                for b_size in range(model.hps.meta_batch_size):
                    # print(b_size)
                    with tf.device('/cpu:0'):
                        shuffle(all_id)
                        cid_list= all_id[:model.hps.batch_size] 
                        train_iter,val_iter = train_data_loader.get_data_loader(cid_list,claim_id,claim_pers,model.hps.batch_size)

                    model.try_run(val_iter,sess,accumulate_grad_op)
                    
                    res  = model.run_step(val_iter, sess, is_train=False, freeze_layer=model.hps.use_pretrain)
                    v_loss, summaries, step = res['loss'], res['summaries'], res['global_step']

                    train_loss_before.append(v_loss)
               
                #update
                    res_val,val_batch_loss =  do_learning_fix_step(model, train_iter, val_iter, sess)
                    print("do learning is done")
                    val_all_loss = tf.add(val_all_loss,val_batch_loss)
                
                    val_loss, summaries_val, step_val = res_val['loss'], res_val['summaries'], res_val['global_step']
                    print("val_loss:",val_loss)
                    train_loss_meta.append(val_loss)
                    batch_loss+=val_loss

                
                    #reset
                    saver_.restore(sess, tf.train.latest_checkpoint('./save'))  
                    print("reset")
                
                print("one batch is done")
                sess.run(apply_grad_op) 
                the_name_model = './save/' + 'model' + str(meta_iteration) + str(epoch_bs)
                model.saver.save(sess,the_name_model,write_meta_graph=False)
                
            print("epoch: {}, before loss：{} ".format(meta_iteration,np.mean(train_loss_before)))    
            print("epoch: {}, after loss：{} ".format(meta_iteration,np.mean(train_loss_meta)))
                
        
            best_loss = 30
            patience = 5
            stop_count = 0
            if meta_iteration % 2 == 0:        
                num_claim_val = len(all_val_id)
                val_loss_before = []
                val_loss_meta = []
                shuffle(all_val_id)

                for i in range(0,80,model.hps.batch_size):
                    with tf.device('/cpu:0'):
                        val_cid_list= all_val_id[i:i+model.hps.batch_size] 
                        valid_train_iter,valid_val_iter = valid_data_loader.get_data_loader(val_cid_list,claim_val_id,claim_val_pers,model.hps.batch_size)
                    
                    res = model.run_step(valid_val_iter, sess, is_train=False, freeze_layer=model.hps.use_pretrain)  
                    loss = res['loss']
                    val_loss_before.append(loss)
                
                    #meta tuning
                    res_val_,val_batch_loss =  do_learning_fix_step(model, valid_train_iter, valid_val_iter, sess)
                    val_loss_meta.append(res_val_['loss']) 
                    saver_.restore(sess, tf.train.latest_checkpoint('./save'))
                    
                print("epoch: {}, fine tuning loss：{} ".format(meta_iteration,np.mean(val_loss_meta)))

                if np.mean(val_loss_meta)< best_loss:
                    best_loss = np.mean(val_loss_meta)
                    the_meta_model = train_savedir + 'MetaModel' + str(meta_iteration)
                    model.saver.save(sess, the_meta_model)  
                    print("save fine tuning model in {}".format(train_savedir))
                else:
                    stop_count+=1
                    if stop_count>patience:
                        print("loss has been rising, stop training")
                        break

def main():
  
    utils.print_config(args)


    if 'train' not in args.mode:
        args.keep_rate = 1.0
    args.use_pretrain = True if args.use_pretrain == 'True' else False
    args.use_aux_task = True if args.use_aux_task == 'True' else False


    if args.mode == 'lm_train':   
        args.model = 'lm'
        args.data_path = "./data/wikitext/wikitext-103/processed_wiki_train.bin"  
        args.use_pretrain = False  

    args.model_path = os.path.join(args.model_path, args.exp_name).format(args.model)  #model_path default="data/log/{}

    if not os.path.exists(args.model_path):
        if 'train' not in args.mode:
            print(args.model_path)
            raise ValueError
        os.makedirs(args.model_path)
    with open(os.path.join(args.model_path, 'config.json'), 'w', encoding='utf8') as f:
        json.dump(vars(args), f)

    print("Default models path: {}".format(args.model_path))

    print('code start/ {} mode / {} models'.format(args.mode, args.model))
    utils.assign_specific_gpu(args.gpu_nums)

    vocab = utils.Vocab() 

    vardicts = utils.get_pretrain_weights(
        args.true_pretrain_ckpt_path) if args.use_pretrain and args.mode == 'train' else None 

    if args.mode == 'decode':
        if args.model == 'mmi_bidi': args.beam_size = args.mmi_bsize
        args.batch_size = args.beam_size

    modelhps = deepcopy(args)
    if modelhps.mode == 'decode':
        modelhps.max_dec_len = 1

    if args.model == 'vanilla':
        model = BaseModel(vocab, modelhps)
    elif args.model == 'mmi_bidi':
        if args.mode == 'decode':
            bw_graph = tf.Graph()
            with bw_graph.as_default():
                bw_model = BaseModel(vocab, args)

            bw_sess = tf.Session(graph=bw_graph, config=utils.gpu_config())

            with bw_sess.as_default():
                with bw_graph.as_default():
                    bidi_ckpt_path = utils.load_ckpt(bw_model.hps, bw_model.saver, bw_sess)

            fw_graph = tf.Graph()
            with fw_graph.as_default():
                modelhps.model_path = modelhps.model_path.replace('mmi_bidi', 'vanilla')
                modelhps.model = 'vanilla'
                fw_model = BaseModel(vocab, modelhps)
            fw_sess = tf.Session(graph=fw_graph)
            with fw_sess.as_default():
                with fw_graph.as_default():
                    ckpt_path = utils.load_ckpt(fw_model.hps, fw_model.saver, fw_sess)
        else:
            model = BaseModel(vocab, modelhps)

    elif args.model == 'lm':
        model = LMModel(vocab, modelhps)  
    elif args.model == 'embmin':
        model = DiverEmbMin(vocab, modelhps)
    else:
        raise ValueError
    print('models load end') 



    if args.mode in ['train', 'lm_train']:
        train(model,vocab, vardicts)
    elif args.mode == 'decode':
        import time

        if args.model == 'mmi_bidi':
            batcher = Batcher(vocab, bw_model.hps.data_path.replace('train_', 'test_'), args)
            decoder = BeamsearchDecoder(fw_model, batcher, vocab, fw_sess=fw_sess, bw_model=bw_model, bw_sess=bw_sess, bidi_ckpt_path=bidi_ckpt_path)
        else:
            batcher = Batcher(vocab, model.hps.data_path.replace('train_', 'test_'), args)
            decoder = BeamsearchDecoder(model, batcher, vocab)
        decoder.decode()
    elif args.mode == 'eval':
        pass

if __name__ == '__main__':
    main()
