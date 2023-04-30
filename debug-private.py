import pandas as pd
import io
import numpy as np
import sys
import json
import torch
from google.cloud import storage, bigquery

from src.models.ast_models import ASTModel_pretrain, ASTModel_finetune
from src.dataloader_gcs import AudioDataset
from src.utilities.speech_utils import *

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

def prep_ssast_data(df,target_labels,save_name,create_label_csv=False):
    data_list=[]
    for i in range(df.shape[0]):
        data_list.append({
            'wav':df.index[i],
            'labels':df.iloc[i][target_labels].values.tolist()
        })
    
    all_labels=[d['labels'] for d in data_list]
    all_str=[('_').join(map(str, l)) for l in all_labels]
    
    label_df=pd.DataFrame({
        'mid':all_labels,
        'display_name':all_str
    })
    
    with open(save_name + '.json', 'w') as f:
        json.dump({'data': data_list}, f, indent=1)
        
    if create_label_csv:
        label_df.drop_duplicates('display_name',inplace=True)
        label_df.reset_index(inplace=True,drop=True)
        label_df.reset_index(inplace=True)
        label_df.to_csv('label_df.csv')


train_df=train_df[target_labels]
test_df=train_df[target_labels]

prep_ssast_data(train_df,target_labels,'train_ssast',create_label_csv=True)
prep_ssast_data(test_df,target_labels,'test_ssast')

dataset_mean=-4.2677393
dataset_std=4.5689974
audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'demo',
              'mode':'train', 'mean':dataset_mean, 'std':dataset_std, 'noise':False}
#new_audio_conf = {'resample_rate':16000, 'reduce': True, 'clip_length':0, 'tshift':0.9, 'speed':0, 'gauss_noise':0.8, 'pshift':0, 'pshiftn':0, 'gain':0.9, 'stretch': 0, 'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'demo',
#              'mode':'train', 'mean':dataset_mean, 'std':dataset_std, 'noise':False}

train_data=AudioDataset('train_ssast.json',audio_conf,bucket,gcs_prefix,label_csv='label_df.csv')
#test_data=AudioDataset('test_ssast.json',audio_conf,bucket,gcs_prefix,label_csv='label_df.csv')

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=1, shuffle=True, num_workers=0)

#test_loader = torch.utils.data.DataLoader(
 #   test_data,
  #  batch_size=8, shuffle=False, num_workers=0)

#test1
batch=next(iter(train_loader))
prediction = ast_mdl(batch[0])
print(prediction.shape)
print(prediction[1,])

for param in ast_mdl.v.parameters():
    param.requires_grad = False
    
model_parameters = filter(lambda p: p.requires_grad, ast_mdl.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f'Number of trainable parameters: {params}')

#run
criterion = torch.nn.BCEWithLogitsLoss()
optim = torch.optim.AdamW([p for p in ast_mdl.parameters() if p.requires_grad])

epochs = 1
for e in range(epochs):
    running_loss = 0
    for i, batch in enumerate(train_loader):
        x = batch[0]
        targets = batch[1]
        optim.zero_grad()
        o =  ast_mdl(batch[0])
        loss = criterion(o, targets)
        loss.backward()
        optim.step()
        running_loss += loss.item()
        print(f'Progress: {round(i/len(train_loader)*100)}%    ',end='\r')
        
    print(e, running_loss/len(train_loader))

torch.save(ast_mdl.state_dict(), 'ast_mdl_base_frame_400_speechfeat_13_adamw_1epoch.pt')
ast_mdl.eval()
all_preds=[]
all_targets=[]
for i, batch in enumerate(test_loader):
    x = batch[0]
    targets = batch[1]
    optim.zero_grad()
    o=ast_mdl(batch[0])
    all_preds.append(o)
    all_targets.append(targets)
    print(f'Progress: {round(i/len(test_loader)*100)}%    ',end='\r')

pred_mat=torch.sigmoid(torch.cat(all_preds)).detach().numpy()
target_mat=torch.cat(all_targets).detach().numpy()

from sklearn.metrics import roc_auc_score, roc_curve
#import matplotlib.pyplot as plt

aucs=roc_auc_score(target_mat, pred_mat, average = None)
print(aucs)
data = [
('Label', target_labels),
('AUC', target_labels)]
pd.DataFrame({'Label':target_labels, 'AUC':aucs})
