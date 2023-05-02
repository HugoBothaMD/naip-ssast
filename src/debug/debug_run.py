#IMPORTS
#built-in
import argparse
import ast
import os
import pickle
import sys
import time
import json

#third party
import numpy as np
import pandas as pd
import torch

from google.cloud import storage, bigquery
from torch.utils.data import WeightedRandomSampler

#local
#from dataloader_mayo import AudioDataset
from dataloader_gcs import AudioDataset
from models import ASTModel_pretrain, ASTModel_finetune
#from traintest_mayo import train, validate
#from traintest_mask_mayo import trainmask
from traintest_mask import trainmask

project_name = 'ml-mps-aif-afdgpet01-p-6827'
study = 'speech_poc_freeze_1'
bucket_name = 'ml-e107-phi-shared-aif-us-p'
gcs_prefix = f'speech_ai/speech_lake/{study}'

storage_client = storage.Client(project=project_name)
bq_client = bigquery.Client(project=project_name)
bucket = storage_client.bucket(bucket_name)

file_list=[]
for blob in storage_client.list_blobs(bucket_name, prefix=gcs_prefix):
    file_list.append(blob.name)

    extensions=[f.split('.')[-1] for f in file_list]

data_split_root = 'gs://ml-e107-phi-shared-aif-us-p/speech_ai/share/data_splits/amr_subject_dedup_594_train_100_test_binarized_v20220620'
gcs_train_path = f'{data_split_root}/train.csv'
gcs_test_path = f'{data_split_root}/test.csv'

# (1) load the train and test files to a df
train_df = pd.read_csv(gcs_train_path, index_col = 'uid')
test_df = pd.read_csv(gcs_test_path, index_col = 'uid')

# (2) alter columns as necessary 
train_df["distortions"]=((train_df["distorted Cs"]+train_df["distorted V"])>0).astype(int)
test_df["distortions"]=((test_df["distorted Cs"]+test_df["distorted V"])>0).astype(int)

# (3) define target labels
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

#audio configuration
dataset = 'retrospeech'
#audio
resample_rate = 16000
reduce = True
clip_length = 0
#audio augmentations
tshift = 0 #time shift
speed = 0
gauss = 0 #amt noise
pshift = 0 #pitch shift
pshiftn = 0 #pitch shift n steps
gain = 0
stretch = 0
#spectrogram
dataset_mean = -4.2677393
dataset_std = 4.5689974
target_length = 1024
num_mel_bins = 128
freqm = 0
timem = 0
mixup = 0
noise = False
#new_audio_conf = {'resample_rate':16000, 'reduce': True, 'clip_length':0, 'tshift':0, 'speed':0, 'gauss_noise':0, 'pshift':0, 'pshiftn':0, 'gain':0, 'stretch': 0, 'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'demo',
#              'mode':'train', 'mean':dataset_mean, 'std':dataset_std, 'noise':False}
#new_audio_conf = {'resample_rate':16000, 'reduce': True, 'clip_length':0, 'tshift':0.9, 'speed':0, 'gauss_noise':0.8, 'pshift':0, 'pshiftn':0, 'gain':0.9, 'stretch': 0, 'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'demo','mode':'train', 'mean':dataset_mean, 'std':dataset_std, 'noise':False}

#train_data = AudioDataset(train_df, target_labels, new_audio_conf, gcs_prefix, bucket)



#train_loader = torch.utils.data.DataLoader(
 #   train_data,
  #  batch_size=1, shuffle=True, num_workers=0)


# train_audio_conf = {'dataset': dataset, 'mode': 'train', 'resample_rate': resample_rate, 'reduce': reduce, 'clip_length': 0,
#                     'tshift':tshift, 'speed':speed, 'gauss_noise':gauss, 'pshift':pshift, 'pshiftn':pshiftn, 'gain':gain, 'stretch': stretch,
#                     'num_mel_bins': num_mel_bins, 'target_length': target_length, 'freqm': freqm, 'timem': timem, 'mixup': mixup, 'noise':noise,
#                     'mean':dataset_mean, 'std':dataset_std}

# eval_audio_conf = {'dataset': dataset, 'mode': 'evaluation', 'resample_rate': resample_rate, 'reduce': reduce, 'clip_length': 0,
#                     'tshift':tshift, 'speed':speed, 'gauss_noise':gauss, 'pshift':pshift, 'pshiftn':pshiftn, 'gain':gain, 'stretch': stretch,
#                     'num_mel_bins': num_mel_bins, 'target_length': target_length, 'freqm': freqm, 'timem': timem, 'mixup': mixup, 'noise':noise,
#                     'mean':dataset_mean, 'std':dataset_std}

# train_dataset = AudioDataset(train_df, target_labels, train_audio_conf, gcs_prefix, bucket) #librosa = True (might need to debug this one)

# test_dataset = AudioDataset(test_df, target_labels, eval_audio_conf, gcs_prefix, bucket)
#optional validation set
train_df=train_df[target_labels]
test_df=train_df[target_labels]

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

prep_ssast_data(train_df,target_labels,'train_ssast',create_label_csv=True)
prep_ssast_data(test_df,target_labels,'test_ssast')

audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'demo',
              'mode':'train', 'mean':dataset_mean, 'std':dataset_std, 'noise':False}
train_data=AudioDataset('train_ssast.json',audio_conf,bucket,gcs_prefix,label_csv='label_df.csv')
test_data=AudioDataset('test_ssast.json',audio_conf,bucket,gcs_prefix,label_csv='label_df.csv')

batch_size = 1
num_workers = 0

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=8, shuffle=True, num_workers=0)

eval_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=8, shuffle=False, num_workers=0)
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=True)

#EVENTUALLY ADD IN OPTIONAL VALIDATION
#eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=True)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#data loading
parser.add_argument('-i','--prefix',default='speech_ai/speech_lake/speech_poc_freeze_1', help='Input directory or location in google cloud storage bucket containing files to load')
parser.add_argument('-d','--data_split_root', default='gs://ml-e107-phi-shared-aif-us-p/speech_ai/share/data_splits/amr_subject_dedup_594_train_100_test_binarized_v20220620', help='path to datasplit csvs. Assumes it points to a directory with a train.csv and test.csv')
parser.add_argument('-l','--label_txt', default='/Users/m144443/Documents/mayo_ssast/src/labels.txt')
parser.add_argument("--n_class", type=int, default=13, help="number of classes")
#GCS
parser.add_argument('-b','--bucket_name', default='ml-e107-phi-shared-aif-us-p', help="google cloud storage bucket name")
parser.add_argument('-p','--project_name', default='ml-mps-aif-afdgpet01-p-6827', help='google cloud platform project name')
#librosa vs torchaudio
parser.add_argument('--lib', default=True, type=bool, help="Specify whether to load using librosa as compared to torch audio")
#output
parser.add_argument('-o',"--exp_dir", type=str, default="/Users/m144443/Documents/mayo_ssast/experiments", help="directory to dump experiments")
#Audio configuration parameters
parser.add_argument("--dataset", default='mayo',type=str, help="the dataset used for training")
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
#Model parameters
parser.add_argument("--task", type=str, default='pretrain_joint', help="pretraining or fine-tuning task", choices=["ft_avgtok", "ft_cls", "pretrain_mpc", "pretrain_mpg", "pretrain_joint"])
parser.add_argument("--fstride", type=int, default=128,help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=2, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument("--fshape", type=int, default=128,help="shape of patch on the frequency dimension")
parser.add_argument("--tshape", type=int, default=2, help="shape of patch on the time dimension")
parser.add_argument('--model_size', default='base',help='the size of AST models', type=str)
#Training parameters
parser.add_argument('--batch_size', default=8, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--num_workers', default=0, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--epochs", type=int, default=1, help="number of maximum training epochs")
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["adamw", "adam"])
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument('--adaptschedule', help='if use adaptive scheduler ', type=ast.literal_eval, default='False')
parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval, default='True')
#fine-tuning parameters
parser.add_argument("--pretrained_mdl_path", type=str, help="the ssl pretrained models path")#, default='/Users/m144443/Documents/mayo_ssast/pretrained_model/SSAST-Base-Frame-400.pth',) #/Users/m144443/Documents/mayo_ssast/pretrained_model/SSAST-Base-Frame-400.pth
parser.add_argument("--freeze",type=bool, default=True, help="Specify whether to freeze original model before fine-tuning")
parser.add_argument("--basic", type=bool, default=True, help="run basic finetuning/metrics rather than altering lr or anything else")
parser.add_argument("--head_lr", type=int, default=1, help="the factor of mlp-head_lr/lr, used in some fine-tuning experiments only")
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics for validation in finetuning", choices=["mAP", "acc"])
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval, default='False')
parser.add_argument("--wa_start", type=int, default=16, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=30, help="which epoch to end weight averaging in finetuning")
#pretraining parameters
parser.add_argument('--mask_patch', help='how many patches to mask (used only for ssl pretraining)', type=int, default=400)
parser.add_argument("--cluster_factor", type=int, default=3, help="mask clutering factor")
parser.add_argument("--epoch_iter", type=int, default=2000, help="for pretraining, how many iterations to verify and save models")
#evaluation
parser.add_argument('--eval_only', type=bool, default=False, help="specify if you want to only run evaluation - use pretrained_mdl_path to specify which model to load for evaluation")
parser.add_argument("-mdl_path", type=str, default='/Users/m144443/Documents/mayo_ssast/experiments/models/audio_model.1.pth', help="if loading an already pre-trained/fine-tuned model")
args = parser.parse_args()

ast_mdl = ASTModel_pretrain(fshape=128, tshape=2, fstride=128, tstride=2, input_fdim=128, input_tdim=128, model_size='base', load_pretrained_mdl_path=None)
task = 'pretrain_joint'
trainmask(ast_mdl, train_loader, None, args)



ast_mdl = ASTModel_finetune(
    task='ft_cls', label_dim=len(target_labels), fshape=128, tshape=2, fstride=128, tstride=2, input_fdim=128, input_tdim=target_length, 
    model_size='base', load_pretrained_mdl_path='/Users/m144443/Documents/mayo_ssast/pretrained_model/SSAST-Base-Frame-400.pth')

#FREEZE THE MODEL (only finetuning classifier head)
for param in ast_mdl.v.parameters():
    param.requires_grad = False
    
model_parameters = filter(lambda p: p.requires_grad, ast_mdl.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(f'Number of trainable parameters: {params}')

criterion = torch.nn.BCEWithLogitsLoss()
optim = torch.optim.AdamW([p for p in ast_mdl.parameters() if p.requires_grad])

epochs = 1 #for now just 1 epoch
for e in range(epochs):
    running_loss = 0
    for i, batch in enumerate(train_loader):
        x = batch[0]
        #x = batch['fbank']
        targets = batch[1]
        #targets = batch['targets'] #have to change to select targets like this
        optim.zero_grad()
        o =  ast_mdl(x) #no need for task + give just fbank
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
for i, batch in enumerate(eval_loader):
    x = batch[0]
    targets = batch[1]
    optim.zero_grad()
    o=ast_mdl(batch[0],task='ft_cls')
    all_preds.append(o)
    all_targets.append(targets)
    print(f'Progress: {round(i/len(eval_loader)*100)}%    ',end='\r')

#simple metrics
pred_mat=torch.sigmoid(torch.cat(all_preds)).detach().numpy()
target_mat=torch.cat(all_targets).detach().numpy()
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

aucs=roc_auc_score(target_mat, pred_mat, average = None)
print(aucs)
data = [
('Label', target_labels),
('AUC', target_labels)]
pd.DataFrame({'Label':target_labels, 'AUC':aucs})
