# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py
'''
SSAST run function 
Can perform pre-training or fine-tuning
Option for fine-tuning only a classification head (freeze rest of model)
Run embedding extraction

Reformated and edited based on code from Yuan Gong (https://github.com/YuanGongND/ssast/tree/main/src/run.py) 

Last modified: 05/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: run.py
'''
#IMPORTS
#built-in
import argparse
import ast
import json
import os
import pickle

#third party
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pyarrow

from google.cloud import storage, bigquery
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import WeightedRandomSampler

#local
from dataloader_mayo import AudioDataset
from models import ASTModel_pretrain, ASTModel_finetune
from utilities import *
from loops import *

def train_ssast(args):
    """
    Run pretraining or finetuning of SSAST
    :param args: dict with all the argument values
    """
    #(1) Load data, note that we are not doing any validation
    assert '.csv' not in args.data_split_root, f'May have given a full file path, please confirm this is a directory: {args.data_split_root}'
    train_df, val_df, test_df = load_data(args.data_split_root, args.exp_dir, args.cloud, args.cloud_dir, args.bucket)

    if args.debug:
        train_df = train_df.iloc[0:8,:]
        val_df = val_df.iloc[0:8,:]
        test_df = test_df.iloc[0:8,:]

    #(2) set audio configurations (val loader and eval loader will both use the eval_audio_conf
    train_audio_conf = {'dataset': args.dataset, 'mode': 'train', 'resample_rate': args.resample_rate, 'reduce': args.reduce, 'clip_length': args.clip_length,
                    'tshift':args.tshift, 'speed':args.speed, 'gauss_noise':args.gauss, 'pshift':args.pshift, 'pshiftn':args.pshiftn, 'gain':args.gain, 'stretch': args.stretch,
                    'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'noise':args.noise,
                    'mean':args.dataset_mean, 'std':args.dataset_std, 'skip_norm':args.skip_norm}
    
    eval_audio_conf = {'dataset': args.dataset, 'mode': 'evaluation', 'resample_rate': args.resample_rate, 'reduce': args.reduce, 'clip_length': args.clip_length,
                    'tshift':args.tshift, 'speed':args.speed, 'gauss_noise':args.gauss, 'pshift':args.pshift, 'pshiftn':args.pshiftn, 'gain':args.gain, 'stretch': args.stretch,
                    'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'noise':args.noise,
                    'mean':args.dataset_mean, 'std':args.dataset_std, 'skip_norm':args.skip_norm}
    
    #(3) Generate audio dataset, note that if bucket not given, it assumes None and loads from local files
    train_dataset = AudioDataset(annotations_df=train_df, target_labels=args.target_labels, audio_conf=train_audio_conf, 
                                 prefix=args.prefix, bucket=args.bucket, librosa=args.lib) #librosa = True (might need to debug this one)
    val_dataset = AudioDataset(annotations_df=val_df, target_labels=args.target_labels, audio_conf=eval_audio_conf, 
                                 prefix=args.prefix, bucket=args.bucket, librosa=args.lib) #librosa = True (might need to debug this one)
    eval_dataset = AudioDataset(annotations_df=test_df, target_labels=args.target_labels, audio_conf=eval_audio_conf, 
                                prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
    
    #(4) set up data loaders (val loader always has batchsize 1)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    print('Now train with {:s} with {:d} training samples, evaluate with {:d} samples'.format(args.dataset, len(train_loader.dataset), len(eval_loader.dataset)))

     #(5) set up model based on specified task and run
    if 'pretrain' in args.task:
        cluster = (args.num_mel_bins != args.fshape)
        if cluster == True:
            print('The num_mel_bins {:d} and fshape {:d} are different, not masking a typical time frame, using cluster masking.'.format(args.num_mel_bins, args.fshape))
        else:
            print('The num_mel_bins {:d} and fshape {:d} are same, masking a typical time frame, not using cluster masking.'.format(args.num_mel_bins, args.fshape))
        #initialize model
        # no label dimension needed as it is self-supervised, fshape=fstride and tshape=tstride
        ast_mdl = ASTModel_pretrain(fshape=args.fshape, tshape=args.tshape,
                                    fstride=args.fshape, tstride=args.tshape,
                                    input_fdim=args.num_mel_bins, input_tdim=args.target_length,
                                    model_size=args.model_size, load_pretrained_mdl_path=None) 
        print('Note that pre-training further from an already pre-trained model is not supported. If it becomes supported, will need to alter the load_pretrained_mdl_path variable in model initalization')
    
        #run
        print('Now starting self-supervised pretraining for {:d} epochs'.format(args.epochs))
        ast_mdl = pretrain(ast_mdl, train_loader, val_loader, 
                           args.optim, args.learning_rate,
                           args.scheduler, args.max_lr,
                           args. epochs,
                           cluster, args.task, args.mask_patch,
                           args.exp_dir, args.cloud, args.cloud_dir, args.bucket)
        
        print('Saving final model')
        mdl_path = os.path.join(args.exp_dir, '{}_{}_{}_{}_epoch{}_ast_pt_mdl.pt'.format(args.dataset,args.model_size, args.n_class, args.optim, args.epochs))
        torch.save(ast_mdl.state_dict(), mdl_path)

        if args.cloud:
            upload(args.cloud_dir, mdl_path, args.bucket)

        print('Training finished')
        print(f'Model saved to: {mdl_path}')

    else:
        #initialize model
        ast_mdl = ASTModel_finetune(task=args.task, label_dim=args.n_class, 
                                    fshape=args.fshape, tshape=args.tshape, 
                                    fstride=args.fstride, tstride=args.tstride,
                                    input_fdim=args.num_mel_bins, input_tdim=args.target_length, 
                                    model_size=args.model_size, load_pretrained_mdl_path=args.pretrained_mdl_path, 
                                    activation=args.activation, final_dropout=args.final_dropout, layernorm=args.layernorm, freeze=args.freeze, 
                                    weighted=args.weighted, layer=args.layer)


        model_parameters = filter(lambda p: p.requires_grad, ast_mdl.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Number of trainable parameters: {params}')

        #run finetuning
        print('Now starting fine-tuning for {:d} epochs'.format(args.epochs))
        ast_mdl = finetune(ast_mdl, train_loader, val_loader,
                           args.optim, args.learning_rate, args.loss, 
                           args.scheduler, args.max_lr, args.epochs,
                           args.exp_dir, args.cloud, args.cloud_dir, args.bucket)

        print('Saving final model')
        mdl_path = os.path.join(args.exp_dir, '{}_{}_{}_{}_epoch{}_ast_ft_mdl.pt'.format(args.dataset,args.model_size, args.n_class, args.optim, args.epochs))
        torch.save(ast_mdl.state_dict(), mdl_path)

        if args.cloud:
            upload(args.cloud_dir, mdl_path, args.bucket)

        #EVALUATE finetuned model
        preds, targets = evaluation(ast_mdl, eval_loader,
                                    args.exp_dir, args.cloud, args.cloud_dir, args.bucket)


def eval_only(args):
    """
    Run only evaluation of a pre-existing model
    :param args: dict with all the argument values
    """
    assert args.finetuned_mdl_path is not None, 'Evaluation must be run on a finetuned model, otherwise classification head is completely untrained.'
    # get original model args (or if no finetuned model, uses your original args)
    model_args, args.finetuned_mdl_path = setup_mdl_args(args)

    #(1) Load data, note that we are not doing any validation
    if '.csv' not in args.data_split_root:
        train_df, val_df, test_df = load_data(args.data_split_root, args.exp_dir, args.cloud, args.cloud_dir, args.bucket)
    else:
        test_df = pd.read_csv(args.data_split_root, index_col = 'uid')
        test_df["distortions"]=((test_df["distorted Cs"]+test_df["distorted V"])>0).astype(int)

    if args.debug:
        test_df = test_df.iloc[0:8,:]

    #(2) set audio configurations
    eval_audio_conf = {'dataset': args.dataset, 'mode': 'evaluation', 'resample_rate': args.resample_rate, 'reduce': args.reduce, 'clip_length': args.clip_length,
                    'tshift':args.tshift, 'speed':args.speed, 'gauss_noise':args.gauss, 'pshift':args.pshift, 'pshiftn':args.pshiftn, 'gain':args.gain, 'stretch': args.stretch,
                    'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'noise':args.noise,
                    'mean':args.dataset_mean, 'std':args.dataset_std, 'skip_norm':args.skip_norm}

    #(3) Generate audio dataset, note that if bucket not given, it assumes None and loads from local files
    eval_dataset = AudioDataset(annotations_df=test_df, target_labels=model_args.target_labels, audio_conf=eval_audio_conf, 
                                prefix=args.prefix, bucket=args.bucket, librosa=args.lib)

    #(4) set up data loaders
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    
    #(5) initialize and load in finetuned model
    ast_mdl = ASTModel_finetune(task=model_args.task, label_dim=model_args.n_class, 
                                fshape=model_args.fshape, tshape=model_args.tshape, 
                                fstride=model_args.fstride, tstride=model_args.tstride,
                                input_fdim=model_args.num_mel_bins, input_tdim=model_args.target_length, 
                                model_size=model_args.model_size, load_pretrained_mdl_path=model_args.pretrained_mdl_path,
                                activation=model_args.activation, final_dropout=model_args.final_dropout, layernorm=model_args.layernorm, freeze=model_args.freeze,
                                weighted=model_args.weighted, layer=model_args.layer)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.finetuned_mdl_path, map_location=device)
    ast_mdl.load_state_dict(sd, strict=False)

    #(6) evaluate
    preds, targets = evaluation(ast_mdl, eval_loader, 
                                args.exp_dir, args.cloud, args.cloud_dir, args.bucket)

def get_embeddings(args):
    """
    Run embedding extraction from start to finish
    :param args: dict with all the argument values
    """
    print('Running Embedding Extraction: ')

    # get original model args (or if no finetuned model, uses your original args)
    model_args, args.finetuned_mdl_path = setup_mdl_args(args)

    # (1) load data to get embeddings for
    assert '.csv' in args.data_split_root, f'A csv file is necessary for embedding extraction. Please make sure this is a full file path: {args.data_split_root}'
    annotations_df = pd.read_csv(args.data_split_root, index_col = 'uid')
    annotations_df["distortions"]=((annotations_df["distorted Cs"]+annotations_df["distorted V"])>0).astype(int)

    if args.debug:
        annotations_df = annotations_df.iloc[0:8,:]

    #(2) set audio configurations
    audio_conf = {'dataset': args.dataset, 'mode': 'evaluation', 'resample_rate': args.resample_rate, 'reduce': args.reduce, 'clip_length': args.clip_length,
                    'tshift':args.tshift, 'speed':args.speed, 'gauss_noise':args.gauss, 'pshift':args.pshift, 'pshiftn':args.pshiftn, 'gain':args.gain, 'stretch': args.stretch,
                    'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'noise':args.noise,
                    'mean':args.dataset_mean, 'std':args.dataset_std, 'skip_norm':args.skip_norm}

    # (3) set up dataloader with current args
    dataset = AudioDataset(annotations_df=annotations_df, target_labels=model_args.target_labels, audio_conf=audio_conf, 
                                prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn) 

    ast_mdl = ASTModel_finetune(task=model_args.task, label_dim=model_args.n_class, 
                                fshape=model_args.fshape, tshape=model_args.tshape, 
                                fstride=model_args.fstride, tstride=model_args.tstride,
                                input_fdim=model_args.num_mel_bins, input_tdim=model_args.target_length, 
                                model_size=model_args.model_size, load_pretrained_mdl_path=model_args.pretrained_mdl_path,
                                activation=model_args.activation, final_dropout=model_args.final_dropout, layernorm=model_args.layernorm, freeze=model_args.freeze,
                                weighted=model_args.weighted, layer=model_args.layer)
    
    if args.finetuned_mdl_path is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(args.finetuned_mdl_path, map_location=device)
        ast_mdl.load_state_dict(sd, strict=False)
    else:
        print(f'Extracting embeddings from only a pretrained model: {args.pretrained_mdl_path}. Extraction method changed to pt.')
        args.embedding_type = 'pt'

    embeddings = embedding_extraction(ast_mdl, loader, args.embedding_type, args.layer, args.task)

    df_embed = pd.DataFrame([[r] for r in embeddings], columns = ['embedding'], index=annotations_df.index)
    
    if args.embedding_type == 'ft':
        args.layer='NA'
        args.task='NA'
    elif args.embedding_type == 'wt':
        args.layer='NA'
    elif args.layer==-1:
        args.layer='Final'
    try:
        if args.finetuned_mdl_path is not None:
            args.finetuned_mdl_path = args.finetuned_mdl_path.replace(os.path.commonprefix([args.dataset, os.path.basename(args.finetuned_mdl_path)]), '')
            pqt_path = '{}/{}_{}_{}_layer{}_{}_ssast_{}_embeddings.pqt'.format(args.exp_dir, args.dataset, os.path.basename(args.finetuned_mdl_path)[:-3], args.model_size, args.layer, args.task, args.embedding_type,)
        else:
            pqt_path = '{}/{}_{}_layer{}_{}_ssast_{}_embeddings.pqt'.format(args.exp_dir, args.dataset, args.model_size, args.layer, args.task,args.embedding_type,)
        df_embed.to_parquet(path=pqt_path, index=True, engine='pyarrow') 

        if args.cloud:
            upload(args.cloud_dir, pqt_path, args.bucket)
    except:
        print('Unable to save as pqt, saving instead as csv')
        if args.finetuned_mdl_path is not None:
            args.finetuned_mdl_path = args.finetuned_mdl_path.replace(os.path.commonprefix([args.dataset, os.path.basename(args.finetuned_mdl_path)]), '')
            csv_path = '{}/{}_{}_{}_layer{}_{}_ssast_{}_embeddings.csv'.format(args.exp_dir, args.dataset, os.path.basename(args.finetuned_mdl_path)[:-3], args.model_size, args.layer, args.task, args.embedding_type,)
        else:
            csv_path = '{}/{}_{}_layer{}_{}_ssast_{}_embeddings.csv'.format(args.exp_dir, args.dataset, args.model_size, args.layer, args.task,args.embedding_type,)
        df_embed.to_csv(csv_path, index=True)

        if args.cloud:
            upload(args.cloud_dir, csv_path, args.bucket)

    print('Embedding extraction finished')
    return df_embed

def main():     
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #Inputs
    parser.add_argument('-i','--prefix',default='speech_ai/speech_lake/', help='Input directory or location in google cloud storage bucket containing files to load')
    parser.add_argument("-s", "--study", choices = ['r01_prelim','speech_poc_freeze_1', None], default='speech_poc_freeze_1', help="specify study name")
    parser.add_argument("-d", "--data_split_root", default='gs://ml-e107-phi-shared-aif-us-p/speech_ai/share/data_splits/amr_subject_dedup_594_train_100_test_binarized_v20220620', help="specify file path where datasplit is located. If you give a full file path to classification, an error will be thrown. On the other hand, evaluation and embedding expects a single .csv file.")
    parser.add_argument('-l','--label_txt', default='/Users/m144443/Documents/GitHub/mayo-ssast/src/labels.txt')
    parser.add_argument('--lib', default=False, type=bool, help="Specify whether to load using librosa as compared to torch audio")
    #GCS
    parser.add_argument('-b','--bucket_name', default='ml-e107-phi-shared-aif-us-p', help="google cloud storage bucket name")
    parser.add_argument('-p','--project_name', default='ml-mps-aif-afdgpet01-p-6827', help='google cloud platform project name')
    parser.add_argument('--cloud', default=False, type=bool, help="Specify whether to save everything to cloud")
    #output
    parser.add_argument("--dataset", default=None,type=str, help="When saving, the dataset arg is used to set file names. If you do not specify, it will assume the lowest directory from data_split_root")
    parser.add_argument("-o", "--exp_dir", default="./debug_exp/weighted", help='specify LOCAL output directory')
    parser.add_argument('--cloud_dir', default='m144443/temp_out/ssast_orig', type=str, help="if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket")
    #Mode specific
    parser.add_argument("-m", "--mode", choices=['train','eval','extraction'], default='train')
    parser.add_argument("--task", type=str, default='ft_cls', help="pretraining or fine-tuning task", choices=["ft_avgtok", "ft_cls", "pretrain_mpc", "pretrain_mpg", "pretrain_joint"])
    parser.add_argument("--pretrained_mdl_path", type=str, default='/Users/m144443/Documents/mayo_ssast/pretrained_model/SSAST-Base-Frame-400.pth', help="the ssl pretrained models path")#, default='/Users/m144443/Documents/mayo_ssast/pretrained_model/SSAST-Base-Frame-400.pth',) #/Users/m144443/Documents/mayo_ssast/pretrained_model/SSAST-Base-Frame-400.pth
    parser.add_argument("--finetuned_mdl_path", type=str, default='/Users/m144443/Documents/GitHub/mayo-ssast/debug_exp/weighted/amr_subject_dedup_594_train_100_test_binarized_v20220620_base_13_adam_epoch1_ast_ft_mdl.pt', help="if loading an already pre-trained/fine-tuned model")
    parser.add_argument("--freeze",type=bool, default=True, help="Specify whether to freeze original model before fine-tuning")
    parser.add_argument("--weighted",type=bool, default=True, help="Specify whether to train the weight sum of layers")
    parser.add_argument("--layer",type=int, default=-1, help="Specify which model layer output to use. Default is -1 which is the final layer.")
    parser.add_argument('--embedding_type', type=str, default='wt', help='specify whether embeddings should be extracted from classification head (ft) or base pretrained model (pt)', choices=['ft','pt'])
    #Audio configuration parameters
    parser.add_argument("--dataset_mean", default=-4.2677393, type=float, help="the dataset mean, used for input normalization")
    parser.add_argument("--dataset_std", default=4.5689974, type=float, help="the dataset std, used for input normalization")
    parser.add_argument("--target_length", default=1024, type=int, help="the input length in frames")
    parser.add_argument("--num_mel_bins", default=128,type=int, help="number of input mel bins")
    parser.add_argument("--resample_rate", default=16000,type=int, help='resample rate for audio files')
    parser.add_argument("--reduce", default=True, type=bool, help="Specify whether to reduce to monochannel")
    parser.add_argument("--clip_length", default=160000, type=int, help="If truncating audio, specify clip length in # of frames. 0 = no truncation")
    parser.add_argument("--tshift", default=0, type=float, help="Specify p for time shift transformation")
    parser.add_argument("--speed", default=0, type=float, help="Specify p for speed tuning")
    parser.add_argument("--gauss", default=0, type=float, help="Specify p for adding gaussian noise")
    parser.add_argument("--pshift", default=0, type=float, help="Specify p for pitch shifting")
    parser.add_argument("--pshiftn", default=0, type=float, help="Specify number of steps for pitch shifting")
    parser.add_argument("--gain", default=0, type=float, help="Specify p for gain")
    parser.add_argument("--stretch", default=0, type=float, help="Specify p for audio stretching")
    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
    parser.add_argument('--timem', help='time mask max length', type=int, default=0)
    parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
    parser.add_argument("--noise", type=bool, default=False, help="specify if augment noise in finetuning")
    parser.add_argument("--skip_norm", type=bool, default=False, help="specify whether to skip normalization on spectrogram")
    #Model parameters
    parser.add_argument("--fstride", type=int, default=128,help="soft split freq stride, overlap=patch_size-stride")
    parser.add_argument("--tstride", type=int, default=2, help="soft split time stride, overlap=patch_size-stride")
    parser.add_argument("--fshape", type=int, default=128,help="shape of patch on the frequency dimension")
    parser.add_argument("--tshape", type=int, default=2, help="shape of patch on the time dimension")
    parser.add_argument('--model_size', default='base',help='the size of AST models', type=str)
    #Training parameters
    parser.add_argument('--batch_size', default=2, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--num_workers', default=0, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
    parser.add_argument("--epochs", type=int, default=1, help="number of maximum training epochs")
    parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
    parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["adamw", "adam"])
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument("--scheduler", type=str, default=None, help="specify lr scheduler", choices=["onecycle", None])
    parser.add_argument("--max_lr", type=float, default=0.01, help="specify max lr for lr scheduler")
    #Pretraining parameters
    parser.add_argument('--mask_patch', help='how many patches to mask (used only for ssl pretraining)', type=int, default=400)
    parser.add_argument("--cluster_factor", type=int, default=3, help="mask clutering factor")
    #classification head parameters
    parser.add_argument("--activation", type=str, default='relu', help="specify activation function to use for classification head")
    parser.add_argument("--final_dropout", type=float, default=0.25, help="specify dropout probability for final dropout layer in classification head")
    parser.add_argument("--layernorm", type=bool, default=True, help="specify whether to include the LayerNorm in classification head")
    #OTHER
    parser.add_argument("--debug", default=True, type=bool)
    args = parser.parse_args()

    #variables
    # (1) Set up GCS
    if args.bucket_name is not None:
        storage_client = storage.Client(project=args.project_name)
        bucket = storage_client.bucket(args.bucket_name)
    else:
        bucket = None

    # (2), check if given study or if the prefix is the full prefix.
    if args.study is not None:
        args.prefix = os.path.join(args.prefix, args.study)
    
    # (3) get dataset name
    if args.dataset is None:
        if '.csv' in args.data_split_root:
            args.dataset = '{}_{}'.format(os.path.basename(os.path.dirname(args.data_split_root)), os.path.basename(args.data_split_root[:-4]))
        else:
            args.dataset = os.path.basename(args.data_split_root)
    
    # (4) get target labels
     #get list of target labels
    with open(args.label_txt) as f:
        target_labels = f.readlines()
    target_labels = [l.strip() for l in target_labels]
    args.target_labels = target_labels

    args.n_class = len(target_labels)

    # (5) check if output directory exists, SHOULD NOT BE A GS:// path
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    
    # (6) check if PRETRAINED MDL is stored in gcs bucket
    if args.pretrained_mdl_path[:5] =='gs://':
        pretrained_mdl_path = args.pretrained_mdl_path[5:].replace(args.bucket_name,'')[1:]
        pretrained_mdl_path = download_model(pretrained_mdl_path, args.exp_dir, bucket)
        args.pretrained_mdl_path = pretrained_mdl_path
    
    # (7) dump arguments
    args_path = "%s/args.pkl" % args.exp_dir
    with open(args_path, "wb") as f:
        pickle.dump(args, f)
    #in case of error, everything is immediately uploaded to the bucket
    if args.cloud:
        upload(args.cloud_dir, args_path, bucket)

    # (8) check that clip length has been set
    if args.clip_length == 0:
        try: 
            assert args.batch_size == 1, 'Not currently compatible with different length wav files unless batch size has been set to 1'
        except:
            args.batch_size = 1

    #(9) add bucket to args
    args.bucket = bucket
    # (10) run model
    print(args.mode)

    if args.mode == 'train':
        train_ssast(args)
    elif args.mode == 'eval':
        eval_only(args)
    elif args.mode == 'extraction':
        df_embed = get_embeddings(args)
        
if __name__ == "__main__":
    main()
