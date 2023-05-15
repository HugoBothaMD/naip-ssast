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
from traintest_mayo import *
from traintest_mask_mayo import *
from utilities import collate_fn

#GCS helper functions
def download_model(gcs_path,outpath, bucket):
    '''
    Download a model from google cloud storage and the args.pkl file located in the same folder(if it exists)

    Inputs:
    :param gcs_path: full file path in the bucket to a pytorch model(no gs://project-name in the path)
    :param outpath: string path to directory where you want the model to be stored
    :param bucket: initialized GCS bucket object
    Outputs:
    :return mdl_path: a string path to the local version of the finetuned model (args.pkl will be in the same folder as this model)
    '''
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    dir_path = os.path.dirname(gcs_path)
    bn = os.path.basename(gcs_path)
    blobs = bucket.list_blobs(prefix=dir_path)
    mdl_path = ''
    for blob in blobs:
        blob_bn = os.path.basename(blob.name)
        if blob_bn == bn:
            destination_uri = '{}/{}'.format(outpath, blob_bn) #download model 
            mdl_path = destination_uri
        elif blob_bn == 'args.pkl':
            destination_uri = '{}/model_args.pkl'.format(outpath) #download args.pkl as model_args.pkl
        else:
            continue #skip any other files
        if not os.path.exists(destination_uri):
            blob.download_to_filename(destination_uri)
   
    return mdl_path

def upload(gcs_prefix, path, bucket):
    '''
    Upload a file to a google cloud storage bucket
    Inputs:
    :param gcs_dir: directory path in the bucket to save file to (no gs://project-name in the path)
    :param path: local string path of the file to upload
    :param bucket: initialized GCS bucket object
    '''
    assert bucket is not None, 'no bucket given for uploading'
    if gcs_prefix is None:
        gcs_prefix = os.path.dirname(path)
    blob = bucket.blob(os.path.join(gcs_prefix, os.path.basename(path)))
    blob.upload_from_filename(path)

#Load functions
def load_args(args):
    '''
    Load in an .pkl file of args
    :param args: dict with all the argument values
    :return model_args: dict with all the argument values from the finetuned model
    '''
    # assumes that the model is saved in the same folder as an args.pkl file 
    folder = os.path.basename(os.path.dirname(args.finetuned_mdl_path))

    if os.path.exists(os.path.join(folder, 'model_args.pkl')): #if downloaded from gcs into the exp dir, it should be saved under mdl_args.pkl to make sure it doesn't overwrite the args.pkl
        with open(os.path.join(folder, 'model_args.pkl'), 'rb') as f:
            model_args = pickle.load(f)
    elif os.path.exists(os.path.join(folder, 'args.pkl')): #if not downloaded and instead stored in a local place, it will be saved as args.pkl
        with open(os.path.join(folder, 'args.pkl'), 'rb') as f:
            model_args = pickle.load(f)
    else: #if there are no saved args
        print('No args.pkl or model_args.pkl stored with the finetuned model. Using the current args for initializing the finetuned model instead.')
        model_args = args
    
    return model_args

def setup_mdl_args(args):
    '''
    Get model args used during finetuning of the specified model
    :param args: dict with all the argument values
    :return model_args: dict with all the argument values from the finetuned model
    :return finetuned_mdl_path: updated finetuned_mdl_path (in case it needed to be downloaded from gcs)
    '''
    #if running a pretrained model only, use the args from this run
    if args.finetuned_mdl_path is None:
        model_args = args
    else:
    #if running a finetuned model
        #(1): check if saved on cloud and load the model and args.pkl
        if args.finetuned_mdl_path[:5] =='gs://':
                mdl_path = args.finetuned_mdl_path[5:].replace(args.bucket_name,'')[1:]
                args.finetuned_mdl_path = download_model(mdl_path, args.exp_dir, args.bucket)
        
        #(2): load the args used for finetuning
        model_args = load_args(args)

        #(3): check if the checkpoint for the finetuned model is downloaded
        if model_args.pretrained_mdl_path[:5] =='gs://': #if checkpoint on cloud
            checkpoint = model_args.pretrained_mdl_path[5:].replace(model_args.bucket_name,'')[1:]
            if model_args.bucket_name != args.bucket_name: #if the bucket is not the same as the current bucket, initialize the bucket for downloading
                if args.bucket_name is not None:
                    storage_client = storage.Client(project=model_args.project_name)
                    model_args.bucket = storage_client.bucket(model_args.bucket_name)
                else:
                    model_args.bucket = None

                checkpoint = download_model(checkpoint, model_args.bucket) #download with the new bucket
            else:
                checkpoint = download_model(checkpoint, args.bucket) #download with the current bucket
            model_args.pretrained_mdl_path = checkpoint #reset the checkpoint path
        else: #load in from local machine, just need to check that the path exists
            assert os.path.exists(model_args.pretrained_mdl_path), f'Current pretrain checkpoint does not exist on local machine: {model_args.pretrained_mdl_path}'

    return model_args, args.finetuned_mdl_path

def load_data(data_split_root, exp_dir, cloud, cloud_dir, bucket):
    """
    Load the train and test data from a directory. Assumes the train and test data will exist in this directory under train.csv and test.csv
    :param data_split_root: specify str path where datasplit csvs are located
    :param exp_dir: specify LOCAL output directory as str
    :param cloud: boolean to specify whether to save everything to google cloud storage
    :param cloud_dir: if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket
    :param bucket: google cloud storage bucket object
    :return train_df, val_df, test_df: loaded dataframes with annotations
    """
    train_path = f'{data_split_root}/train.csv'
    test_path = f'{data_split_root}/test.csv'
    #get data
    train_df = pd.read_csv(train_path, index_col = 'uid')
    test_df = pd.read_csv(test_path, index_col = 'uid')

    #randomly sample to get validation set 
    val_df = train_df.sample(50)
    train_df = train_df.drop(val_df.index)

    #save validation set
    val_path = os.path.join(exp_dir, 'validation.csv')
    val_df.to_csv(val_path, index=True)

    if cloud:
        upload(cloud_dir, val_path, bucket)

    #alter data columns
    train_df["distortions"]=((train_df["distorted Cs"]+train_df["distorted V"])>0).astype(int)
    val_df["distortions"]=((val_df["distorted Cs"]+val_df["distorted V"])>0).astype(int)
    test_df["distortions"]=((test_df["distorted Cs"]+test_df["distorted V"])>0).astype(int)

    return train_df, val_df, test_df

# model loops
def finetune_train_loop(args, model, dataloader_train, dataloader_val = None):
    """
    Training loop for finetuning SSAST 
    :param args: dict with all the argument values
    :param model: SSAST model
    :param dataloader_train: dataloader object with training data
    :param dataloader_val: dataloader object with validation data
    :return model: fine-tuned SSAST model
    """
    print('Training start')
    #send to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #loss
    if args.loss == 'MSE':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'BCE':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError('MSE must be given for loss parameter')
    #optimizer
    if args.optim == 'adam':
        optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate)
    elif args.optim == 'adamw':
         optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    else:
        raise ValueError('adam must be given for optimizer parameter')
    
    if args.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.max_lr, steps_per_epoch=len(dataloader_train), epochs=args.epochs)
    else:
        scheduler = None
    
    #train
    for e in range(args.epochs):
        training_loss = list()
        #t0 = time.time()
        model.train()
        for batch in tqdm(dataloader_train):
            x = batch['fbank']
            targets = batch['targets']
            x, targets = x.to(device), targets.to(device)
            optim.zero_grad()
            o = model(x)
            loss = criterion(o, targets)
            loss.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()
            loss_item = loss.item()
            training_loss.append(loss_item)

        if e % 10 == 0:
            #SET UP LOGS
            if scheduler is not None:
                lr = scheduler.get_last_lr()
            else:
                lr = args.learning_rate
            logs = {'epoch': e, 'optim':args.optim, 'loss_fn': args.loss, 'lr': lr}
    
            logs['training_loss_list'] = training_loss
            training_loss = np.array(training_loss)
            logs['running_loss'] = np.sum(training_loss)
            logs['training_loss'] = np.mean(training_loss)

            print('RUNNING LOSS', e, np.sum(training_loss) )
            print(f'Training loss: {np.mean(training_loss)}')

            if dataloader_val is not None:
                print("Validation start")
                validation_loss = val_loop(model, criterion, dataloader_val)

                logs['val_loss_list'] = validation_loss
                validation_loss = np.array(validation_loss)
                logs['val_running_loss'] = np.sum(validation_loss)
                logs['val_loss'] = np.mean(validation_loss)
                
                print('RUNNING VALIDATION LOSS',e, np.sum(validation_loss) )
                print(f'Validation loss: {np.mean(validation_loss)}')
            
            #SAVE LOGS
            json_string = json.dumps(logs)
            logs_path = os.path.join(args.exp_dir, 'logs_epoch{}.json'.format(e))
            with open(logs_path, 'w') as outfile:
                json.dump(json_string, outfile)
            
            #SAVE CURRENT MODEL
            mdl_path = os.path.join(args.exp_dir, 'ast_mdl_epoch{}.pt'.format(e))
            torch.save(model.state_dict(), mdl_path)
            
            if args.cloud:
                upload(args.cloud_dir, logs_path, args.bucket)
                #upload_from_memory(model.state_dict(), args.cloud_dir, mdl_path, args.bucket)
                upload(args.cloud_dir, mdl_path, args.bucket)

    print('Training finished')
    mdl_path = os.path.join(args.exp_dir, '{}_{}_{}_{}_epoch{}_ast_mdl.pt'.format(args.dataset,args.model_size, args.n_class, args.optim, args.epochs))
    torch.save(model.state_dict(), mdl_path)

    if args.cloud:
        upload(args.cloud_dir, mdl_path, args.bucket)

    return model

def val_loop(model, criterion, dataloader_val):
    '''
    Validation loop for finetuning the w2v2 classification head. 
    :param model: W2V2 model
    :param criterion: loss function
    :param dataloader_val: dataloader object with validation data
    :return validation_loss: list with validation loss for each batch
    '''
    validation_loss = list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader_val):
            x = batch['fbank']
            targets = batch['targets']
            x, targets = x.to(device), targets.to(device)
            o = model(x)
            val_loss = criterion(o, targets)
            validation_loss.append(val_loss.item())

    return validation_loss

def eval_loop(model, dataloader_eval, exp_dir, cloud=False, cloud_dir=None, bucket=None):
    """
    Start model evaluation
    :param model: SSAST model
    :param dataloader_eval: dataloader object with evaluation data
    :param exp_dir: specify LOCAL output directory as str
    :param cloud: boolean to specify whether to save everything to google cloud storage
    :param cloud_dir: if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket
    :param bucket: google cloud storage bucket object
    :return preds: model predictions
    :return targets: model targets (actual values)
    """
    print('Evaluation start')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs = []
    t = []
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader_eval):
            x = batch['fbank']
            x = x.to(device)
            targets = batch['targets']
            targets = targets.to(device)
            o = model(x)
            outputs.append(o)
            t.append(targets)

    outputs = torch.cat(outputs).cpu().detach()
    t = torch.cat(t).cpu().detach()
    # SAVE PREDICTIONS AND TARGETS 
    pred_path = os.path.join(exp_dir, 'ast_eval_predictions.pt')
    target_path = os.path.join(exp_dir, 'ast_eval_targets.pt')
    torch.save(outputs, pred_path)
    torch.save(t, target_path)

    if cloud:
        upload(cloud_dir, pred_path, bucket)
        upload(cloud_dir, target_path, bucket)

    print('Evaluation finished')
    return outputs, t

def embedding_loop(model, dataloader, embedding_type='ft'):
    """
    Run a specific subtype of evaluation for getting embeddings.
    :param model: W2V2 model
    :param dataloader_eval: dataloader object with data to get embeddings for
    :param embedding_type: string specifying whether embeddings should be extracted from classification head (ft) or base pretrained model (pt)
    :return embeddings: an np array containing the embeddings
    """

    print('Calculating Embeddings')
    embeddings = np.array([])
    # send to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader):
            x = batch['fbank']
            x = x.to(device)
            e = model.extract_embeddings(x, embedding_type)
            if embeddings.size == 0:
                embeddings = e
            else:
                embeddings = np.append(embeddings, e, axis=0)

    return embeddings

def metrics(args, preds, targets):
    """
    Get AUC scores, doesn't return, just saves the metrics to a csv
    :param args: dict with all the argument values
    :param preds: model predictions
    :param targets: model targets (actual values)
    """
    #get AUC score and all data for ROC curve
    metrics = {}
    pred_mat=torch.sigmoid(preds).numpy()
    target_mat=targets.numpy()
    aucs=roc_auc_score(target_mat, pred_mat, average = None) #TODO: this doesn't work when there is an array with all labels as 0???
    print(aucs)
    data = pd.DataFrame({'Label':args.target_labels, 'AUC':aucs})
    data.to_csv(os.path.join(args.exp_dir, 'aucs.csv'), index=False)
    if args.cloud:
        upload(args.cloud_dir, os.path.join(args.exp_dir, 'aucs.csv'), args.bucket)

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

     #(5) set up model based on specified task
    if 'pretrain' in args.task:
        cluster = (args.num_mel_bins != args.fshape)
        if cluster == True:
            print('The num_mel_bins {:d} and fshape {:d} are different, not masking a typical time frame, using cluster masking.'.format(args.num_mel_bins, args.fshape))
        else:
            print('The num_mel_bins {:d} and fshape {:d} are same, masking a typical time frame, not using cluster masking.'.format(args.num_mel_bins, args.fshape))
        #PRETRAIN
        # no label dimension needed as it is self-supervised, fshape=fstride and tshape=tstride
        ast_mdl = ASTModel_pretrain(fshape=args.fshape, tshape=args.tshape,
                                    fstride=args.fshape, tstride=args.tshape,
                                    input_fdim=args.num_mel_bins, input_tdim=args.target_length,
                                    model_size=args.model_size, load_pretrained_mdl_path=None) 
        print('Note that pre-training further from an already pre-trained model is not supported. If it becomes supported, will need to alter the load_pretrained_mdl_path variable in model initalization')
    else:
        ast_mdl = ASTModel_finetune(task=args.task, label_dim=args.n_class, 
                                    fshape=args.fshape, tshape=args.tshape, 
                                    fstride=args.fstride, tstride=args.tstride,
                                    input_fdim=args.num_mel_bins, input_tdim=args.target_length, 
                                    model_size=args.model_size, load_pretrained_mdl_path=args.pretrained_mdl_path, 
                                    activation='relu', final_dropout=0.2, layernorm=True, freeze=args.freeze, pooling_mode=args.pooling_mode)


        model_parameters = filter(lambda p: p.requires_grad, ast_mdl.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Number of trainable parameters: {params}')
    

    #(7) Run models
    if 'pretrain' in args.task:
        print('Now starting self-supervised pretraining for {:d} epochs'.format(args.epochs))
        ast_mdl = trainmask(args=args, audio_model=ast_mdl, train_loader=train_loader, eval_loader=val_loader)
        return ast_mdl
    else:
        print('Now starting fine-tuning for {:d} epochs'.format(args.epochs))
        if not args.original_fn:
            ast_mdl = finetune_train_loop(args, ast_mdl, train_loader, val_loader)
        else:
            ast_mdl = train(args=args, audio_model=ast_mdl, train_loader=train_loader, val_loader=val_loader)
    
    #(8) evaluation:
    if not args.original_fn:
        preds, targets = eval_loop(ast_mdl, eval_loader, args.exp_dir, args.cloud, args.cloud_dir, args.bucket)
        #aucs = metrics(args, preds, targets)
    else:
        evaluation(args=args, audio_model=ast_mdl, eval_loader=eval_loader, val_loader=None) #TODO: does this need val dataloader????
    

def eval_only(args):
    """
    Run only evaluation of a pre-existing model
    :param args: dict with all the argument values
    """
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
 
    ast_mdl = ASTModel_finetune(task=model_args.task, label_dim=model_args.n_class, 
                                fshape=model_args.fshape, tshape=model_args.tshape, 
                                fstride=model_args.fstride, tstride=model_args.tstride,
                                input_fdim=model_args.num_mel_bins, input_tdim=model_args.target_length, 
                                model_size=model_args.model_size, load_pretrained_mdl_path=model_args.pretrained_mdl_path,
                                activation='relu', final_dropout=0.2, layernorm=True, freeze=model_args.freeze, pooling_mode=model_args.pooling_mode)
    
    if args.finetuned_mdl_path is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(args.finetuned_mdl_path, map_location=device)
        ast_mdl.load_state_dict(sd, strict=False)
    else:
        print(f'Evaluating only a pretrained model: {args.pretrained_mdl_path}')

    if not args.original_fn:
        preds, targets = eval_loop(ast_mdl, eval_loader, args.exp_dir, args.cloud, args.cloud_dir, args.bucket)
        #aucs = metrics(args, preds, targets)
    else:
        evaluation(args=args, audio_model=ast_mdl, eval_loader=eval_loader, val_loader=None) #TODO: does this need val dataloader????


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
                                activation='relu', final_dropout=0.2, layernorm=True, freeze=model_args.freeze, pooling_mode=model_args.pooling_mode)
    
    if args.finetuned_mdl_path is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(args.finetuned_mdl_path, map_location=device)
        ast_mdl.load_state_dict(sd, strict=False)
    else:
        print(f'Extracting embeddings from only a pretrained model: {args.pretrained_mdl_path}. Extraction method changed to pt.')
        args.embedding_type = 'pt'

    embeddings = embedding_loop(ast_mdl, loader, args.embedding_type)

    df_embed = pd.DataFrame([[r] for r in embeddings], columns = ['embedding'], index=annotations_df.index)
    
    try:
        if args.finetuned_mdl_path is not None:
            args.finetuned_mdl_path = args.finetuned_mdl_path.replace(os.path.commonprefix([args.dataset, os.path.basename(args.finetuned_mdl_path)]), '')
            pqt_path = '{}/{}_{}_{}_{}_embeddings.pqt'.format(args.exp_dir, args.dataset, os.path.basename(args.finetuned_mdl_path)[:-3], args.model_size, args.embedding_type)
        else:
            pqt_path = '{}/{}_ssast_{}_{}_embeddings.pqt'.format(args.exp_dir, args.dataset, args.model_size, args.embedding_type)
        df_embed.to_parquet(path=pqt_path, index=True, engine='pyarrow') 

        if args.cloud:
            upload(args.cloud_dir, pqt_path, args.bucket)
    except:
        print('Unable to save as pqt, saving instead as csv')
        if args.finetuned_mdl_path is not None:
            args.finetuned_mdl_path = args.finetuned_mdl_path.replace(os.path.commonprefix([args.dataset, os.path.basename(args.finetuned_mdl_path)]), '')
            csv_path = '{}/{}_{}_{}_{}_embeddings.csv'.format(args.exp_dir, args.dataset, os.path.basename(args.finetuned_mdl_path)[:-3], args.model_size, args.embedding_type)
        else:
            csv_path = '{}/{}_ssast_{}_{}_embeddings.csv'.format(args.exp_dir, args.dataset, args.model_size, args.embedding_type)
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
    parser.add_argument("-d", "--data_split_root", default='gs://ml-e107-phi-shared-aif-us-p/speech_ai/share/data_splits/amr_subject_dedup_594_train_100_test_binarized_v20220620/test.csv', help="specify file path where datasplit is located. If you give a full file path to classification, an error will be thrown. On the other hand, evaluation and embedding expects a single .csv file.")
    parser.add_argument('-l','--label_txt', default='/Users/m144443/Documents/GitHub/mayo-ssast/src/labels.txt')
    parser.add_argument('--lib', default=False, type=bool, help="Specify whether to load using librosa as compared to torch audio")
    #GCS
    parser.add_argument('-b','--bucket_name', default='ml-e107-phi-shared-aif-us-p', help="google cloud storage bucket name")
    parser.add_argument('-p','--project_name', default='ml-mps-aif-afdgpet01-p-6827', help='google cloud platform project name')
    parser.add_argument('--cloud', default=False, type=bool, help="Specify whether to save everything to cloud")
    #output
    parser.add_argument("--dataset", default=None,type=str, help="When saving, the dataset arg is used to set file names. If you do not specify, it will assume the lowest directory from data_split_root")
    parser.add_argument("-o", "--exp_dir", default="./experiments2", help='specify LOCAL output directory')
    parser.add_argument('--cloud_dir', default='m144443/temp_out/ssast2', type=str, help="if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket")
    #Mode specific
    parser.add_argument("-m", "--mode", choices=['train','eval','extraction'], default='extraction')
    parser.add_argument("--finetuned_mdl_path", type=str, default='/Users/m144443/Documents/GitHub/mayo-ssast/experiments2/amr_subject_dedup_594_train_100_test_binarized_v20220620_base_13_adam_epoch1_ast_mdl.pt', help="if loading an already pre-trained/fine-tuned model")
    parser.add_argument("--pretrained_mdl_path", type=str, default='/Users/m144443/Documents/mayo_ssast/pretrained_model/SSAST-Base-Frame-400.pth', help="the ssl pretrained models path")#, default='/Users/m144443/Documents/mayo_ssast/pretrained_model/SSAST-Base-Frame-400.pth',) #/Users/m144443/Documents/mayo_ssast/pretrained_model/SSAST-Base-Frame-400.pth
    parser.add_argument("--freeze",type=bool, default=True, help="Specify whether to freeze original model before fine-tuning")
    parser.add_argument('--original_fn', type=bool, default=False, help="specify whether to use the original SSAST functions")
    parser.add_argument('--embedding_type', type=str, default='pt', help='specify whether embeddings should be extracted from classification head (ft) or base pretrained model (pt)', choices=['ft','pt'])
    #Audio configuration parameters
    parser.add_argument("--dataset_mean", default=-4.2677393, type=float, help="the dataset mean, used for input normalization")
    parser.add_argument("--dataset_std", default=4.5689974, type=float, help="the dataset std, used for input normalization")
    parser.add_argument("--target_length", default=1024, type=int, help="the input length in frames")
    parser.add_argument("--num_mel_bins", default=128,type=int, help="number of input mel bins")
    parser.add_argument("--resample_rate", default=16000,type=int, help='resample rate for audio files')
    parser.add_argument("--reduce", default=True, type=bool, help="Specify whether to reduce to monochannel")
    parser.add_argument("--clip_length", default=0, type=int, help="If truncating audio, specify clip length in # of frames. 0 = no truncation")
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
    parser.add_argument("--task", type=str, default='ft_cls', help="pretraining or fine-tuning task", choices=["ft_avgtok", "ft_cls", "pretrain_mpc", "pretrain_mpg", "pretrain_joint"])
    parser.add_argument("--fstride", type=int, default=128,help="soft split freq stride, overlap=patch_size-stride")
    parser.add_argument("--tstride", type=int, default=2, help="soft split time stride, overlap=patch_size-stride")
    parser.add_argument("--fshape", type=int, default=128,help="shape of patch on the frequency dimension")
    parser.add_argument("--tshape", type=int, default=2, help="shape of patch on the time dimension")
    parser.add_argument('--model_size', default='base',help='the size of AST models', type=str)
    parser.add_argument("-pm", "--pooling_mode", default="mean", help="specify method of pooling last hidden layer", choices=['mean','sum','max'])
    #Training parameters
    parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--num_workers', default=0, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
    parser.add_argument("--epochs", type=int, default=1, help="number of maximum training epochs")
    parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
    parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["adamw", "adam"])
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument("--scheduler", type=str, default=None, help="specify lr scheduler", choices=["onecycle", None])
    parser.add_argument("--max_lr", type=float, default=0.01, help="specify max lr for lr scheduler")
    #training parameters, original fn
    parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='False')
    parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
    parser.add_argument('--adaptschedule', help='if use adaptive scheduler ', type=ast.literal_eval, default='False')
    parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
    parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval, default='True')
    parser.add_argument("--head_lr", type=int, default=1, help="the factor of mlp-head_lr/lr, used in some fine-tuning experiments only")
    #original finetuning
    parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
    parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
    parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
    parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval, default='False')
    parser.add_argument("--wa_start", type=int, default=16, help="which epoch to start weight averaging in finetuning")
    parser.add_argument("--wa_end", type=int, default=30, help="which epoch to end weight averaging in finetuning")
    parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics for validation in finetuning", choices=["mAP", "acc"])
    #original pretraining
    parser.add_argument('--mask_patch', help='how many patches to mask (used only for ssl pretraining)', type=int, default=400)
    parser.add_argument("--cluster_factor", type=int, default=3, help="mask clutering factor")
    parser.add_argument("--epoch_iter", type=int, default=2000, help="for pretraining, how many iterations to verify and save models")
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

    #(8) add bucket to args
    args.bucket = bucket
    # (9) run model
    print(args.mode)

    if args.mode == 'train':
        train_ssast(args)
    if args.mode == 'eval':
        eval_only(args)
    if args.mode == 'extraction':
        df_embed = get_embeddings(args)
        
if __name__ == "__main__":
    main()
