'''
Function to get embeddings from SSAST

Last modified: 05/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: run_mayo.py
'''

#IMPORTS
#built-in
import argparse
import os
import pickle

#third party
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from google.cloud import storage, bigquery

#local
from dataloader_mayo import AudioDataset
from models import ASTModel_finetune
from utilities.dataloader_utils import collate_fn

def get_ssast_embeddings(df, target_labels, args, bucket):
    # (1) To get embeddings, first load the arguments used for fine-tuning the model
    args_path = os.path.join(args.exp_dir, 'args.pkl') 
    model_path = os.path.join(args.exp_dir, args.model_name)

    # (2) set up audio conformer with original model args
    with open(args_path, 'rb') as f:
        model_args = pickle.load(f)
    
    #TEMP FIX
    print('delete fix later')
    #the model we debugged with was from a version without skip_norm, and when we were attempting pretraining. Needed to temporarily fix. 
    model_args.skip_norm = False
    model_args.task = 'ft_cls'
    model_args.pretrained_mdl_path = '/Users/m144443/Documents/GitHub/mayo-ssast/pretrained_model/SSAST-Base-Frame-400.pth'
    #model_args.pretrained_mdl_path = '/Users/m144443/Documents/GitHub/mayo-ssast/temp_out/ast_mdl_base_mayo_13_adamw_1epoch.pt'

    audio_conf = {'dataset': model_args.dataset, 'mode': 'evaluation', 'resample_rate': model_args.resample_rate, 'reduce': model_args.reduce, 'clip_length': model_args.clip_length,
                    'tshift':model_args.tshift, 'speed':model_args.speed, 'gauss_noise':model_args.gauss, 'pshift':model_args.pshift, 'pshiftn':model_args.pshiftn, 'gain':model_args.gain, 'stretch': model_args.stretch,
                    'num_mel_bins': model_args.num_mel_bins, 'target_length': model_args.target_length, 'freqm': model_args.freqm, 'timem': model_args.timem, 'mixup': model_args.mixup, 'noise':model_args.noise,
                    'mean':model_args.dataset_mean, 'std':model_args.dataset_std, 'skip_norm':model_args.skip_norm}
    
    # (3) set up dataloader with current args
    dataset = AudioDataset(annotations_df=df, target_labels=target_labels, audio_conf=audio_conf, 
                                prefix=args.prefix, bucket=bucket, librosa=args.lib)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn) 
    
    # (4) load AST model with original model parameters + pre-trained/fine-tuned model
    ast_mdl = ASTModel_finetune(task=model_args.task, label_dim=model_args.n_class, 
                                    fshape=model_args.fshape, tshape=model_args.tshape, 
                                    fstride=model_args.fstride, tstride=model_args.tstride,
                                    input_fdim=model_args.num_mel_bins, input_tdim=model_args.target_length, 
                                    model_size=model_args.model_size, load_pretrained_mdl_path=model_args.pretrained_mdl_path)
 
    ast_mdl.eval()
    #load fine-tuned model
    ast_mdl.load_state_dict(torch.load(model_path))
    
    # (5) get the embedding layer
    activation = {}
    def _get_activation(name):
        def _hook(model, input, output):
            activation[name] = output.detach()
        return _hook
    ast_mdl.mlp_head[0].register_forward_hook(_get_activation('embeddings'))
    
    # (6) Calculate embeddings
    print('Calculating Embeddings')
    all_names=[]
    all_embeddings=[]
    for batch in tqdm(loader):
        logits=ast_mdl(batch['fbank'])
        names = batch['uid']
        #index = 
        all_names.append(names)
        all_embeddings.append(activation['embeddings'].detach().numpy())
        
    all_names=np.concatenate(all_names)
    all_embeddings=np.concatenate(all_embeddings)
    
    embedding_df=pd.DataFrame(all_embeddings).set_index(all_names)
    embedding_df.columns=[str(s) for s in embedding_df.columns]
    
    return embedding_df




def run(args):
    '''
    '''
    #(1) Set up bucket
    if args.bucket_name is not None:
        storage_client = storage.Client(project=args.project_name)
        bq_client = bigquery.Client(project=args.project_name)
        bucket = storage_client.bucket(args.bucket_name)
    else:
        bucket = None
    
    #(2) Load target labels
    #get list of target labels
    with open(args.label_txt) as f:
        target_labels = f.readlines()
    target_labels = [l.strip() for l in target_labels]
        
    args.n_class = len(target_labels)

    #(3) Load data
    test_df = pd.read_csv(args.data_csv, index_col = 'uid')
    test_df["distortions"]=((test_df["distorted Cs"]+test_df["distorted V"])>0).astype(int)

    #(4) get embeddings, we will only be using the test_df
    embeddings_df = get_ssast_embeddings(test_df, target_labels, args, bucket)

    # (5) save embeddings
    out_name = os.path.join(args.model_dir, 'embeddings.csv')
    embeddings_df.to_csv(out_name)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #data loading
    parser.add_argument('-d','--data_csv', default='gs://ml-e107-phi-shared-aif-us-p/speech_ai/share/data_splits/amr_subject_dedup_594_train_100_test_binarized_v20220620/test.csv', help='path to data csv. Assumes it points to a csv file.')
    parser.add_argument('-i','--prefix',default='speech_ai/speech_lake/speech_poc_freeze_1', help='Input directory or location in google cloud storage bucket containing audio files to load')
    parser.add_argument('-l','--label_txt', default='/Users/m144443/Documents/mayo_ssast/src/labels.txt')
    parser.add_argument("--n_class", type=int, default=13, help="number of classes")
    #GCS
    parser.add_argument('-b','--bucket_name', default='ml-e107-phi-shared-aif-us-p', help="google cloud storage bucket name")
    parser.add_argument('-p','--project_name', default='ml-mps-aif-afdgpet01-p-6827', help='google cloud platform project name')
    #librosa vs torchaudio
    parser.add_argument('--lib', default=True, type=bool, help="Specify whether to load using librosa as compared to torch audio")
    #output
    parser.add_argument('-o',"--exp_dir", type=str, default="/Users/m144443/Documents/GitHub/mayo-ssast/temp_out", help="directory with model + associated files")
    parser.add_argument('-mn',"--model_name", type=str, default="ast_mdl_base_mayo_13_adamw_1epoch.pt", help="directory with model + associated files" )
    #embedding batch size
    parser.add_argument('-bs',"--batch_size", type=int, default=8, help="batch size for embeddings")
    parser.add_argument('--num_workers', default=0, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
    args = parser.parse_args()

    if args.num_workers > 0:
        print('If loading from dataloader fails, it may be due to an error with num_workers. Please set to 0 and try again (Error not yet solved)')
    run(args)
        
if __name__ == "__main__":
    main()
