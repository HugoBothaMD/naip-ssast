import pandas as pd
import io
import numpy as np
import sys
import json
import torch
from google.cloud import storage, bigquery

from models.ast_models import ASTModel_pretrain, ASTModel_finetune
from dataloader_mayo import AudioDataset
#from src.utilities.speech_utils import *

project_name = 'ml-mps-aif-afdgpet01-p-6827'
study = 'speech_poc_freeze_1'
bucket_name = 'ml-e107-phi-shared-aif-us-p'
gcs_prefix = f'speech_ai/speech_lake/{study}'

storage_client = storage.Client(project=project_name)
bq_client = bigquery.Client(project=project_name)
bucket = storage_client.bucket(bucket_name)

file_list=[]
for blob in storage_client.list_blobs(bucket_name, prefix='speech_ai/speech_lake/speech_poc_freeze_1'):
    file_list.append(blob.name)

    extensions=[f.split('.')[-1] for f in file_list]

data_split_root = 'gs://ml-e107-phi-shared-aif-us-p/speech_ai/share/data_splits/amr_subject_dedup_594_train_100_test_binarized_v20220620'
gcs_train_path = f'{data_split_root}/train.csv'
gcs_test_path = f'{data_split_root}/test.csv'

train_df = pd.read_csv(gcs_train_path, index_col = 'uid')
test_df = pd.read_csv(gcs_test_path, index_col = 'uid')

train_df["distortions"]=((train_df["distorted Cs"]+train_df["distorted V"])>0).astype(int)
test_df["distortions"]=((test_df["distorted Cs"]+test_df["distorted V"])>0).astype(int)

target_labels=['breathy',
             'loudness decay',
             'slow rate',
             'high pitch',
             'hoarse / harsh',
             'irregular artic breakdowns',
             'rapid rate',
             'reduced OA loudness',
             'abn pitch variability',
             'strained',
             'hypernasal',
             'abn loudness variability',
              'distortions']

input_tdim = 1024

ast_mdl = ASTModel_finetune(task='ft_cls',
    label_dim=len(target_labels),
    fshape=128, tshape=2, fstride=128, tstride=2,input_fdim=128, 
    input_tdim=input_tdim, model_size='base',
    load_pretrained_mdl_path='./pretrained_model/SSAST-Base-Frame-400.pth'
)

train_df2=train_df[target_labels]
test_df=train_df[target_labels]

dataset_mean=-4.2677393
dataset_std=4.5689974
#audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'demo',
      #        'mode':'train', 'mean':dataset_mean, 'std':dataset_std, 'noise':False}
new_audio_conf = {'resample_rate':16000, 'reduce': True, 'clip_length':0, 'tshift':0, 'speed':0, 'gauss_noise':0, 'pshift':0, 'pshiftn':0, 'gain':0, 'stretch': 0, 'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'demo',
              'mode':'train', 'mean':dataset_mean, 'std':dataset_std, 'noise':False}
#new_audio_conf = {'resample_rate':16000, 'reduce': True, 'clip_length':0, 'tshift':0.9, 'speed':0, 'gauss_noise':0.8, 'pshift':0, 'pshiftn':0, 'gain':0.9, 'stretch': 0, 'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'demo','mode':'train', 'mean':dataset_mean, 'std':dataset_std, 'noise':False}

train_data = AudioDataset(train_df, target_labels, new_audio_conf, gcs_prefix, bucket, True)
#train_data=AudioDataset('train_ssast.json',new_audio_conf,bucket,gcs_prefix,label_csv='label_df.csv')
#test_data=AudioDataset('test_ssast.json',audio_conf,bucket,gcs_prefix,label_csv='label_df.csv')

train_data2 = AudioDataset(train_df2, target_labels, new_audio_conf, gcs_prefix, bucket)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=1, shuffle=True, num_workers=0)

train_loader2 = torch.utils.data.DataLoader(
    train_data2,
    batch_size=1, shuffle=True, num_workers=0)

#test_loader = torch.utils.data.DataLoader(
 #   test_data,
  #  batch_size=8, shuffle=False, num_workers=0)

#test1
batch=next(iter(train_loader))

print('batch finished')