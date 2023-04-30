import pandas as pd
import io
import numpy as np
import json
import argparse
import torch
from tqdm import tqdm

from src.models.ast_models import ASTModel_finetune

from src.dataloader_gcs import AudioDataset

from google.cloud import storage, bigquery


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

def load_model(args,model_name, pretrain_path):
    ast_mdl = ASTModel_finetune(task='ft_cls',
        label_dim=6,
        fshape=128, tshape=2, fstride=128, tstride=2,input_fdim=128, 
        input_tdim=args.input_tdim, model_size=args.model_size.lower(),
        load_pretrained_mdl_path=pretrain_path
    )
    
    ast_mdl.eval()
    
    ast_mdl.load_state_dict(torch.load(f'./models/{model_name}.pt'))
    
    return ast_mdl


def get_dataloader(audio_conf,bucket,gcs_prefix,args):
    
    test_data=AudioDataset(
        'test_ssast.json',audio_conf,bucket,gcs_prefix,label_csv='label_df.csv',
        resample=args.resample,resample_rate=args.resample_rate
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8
    )
    
    return test_loader


def get_ssast_embeddings(model_name,args, bucket,gcs_prefix):
    
    f = open(f'models/{model_name}.json')
    data = json.load(f)
    args=pd.Series(data['arguments'])
    args.batch_size=32
    
    audio_conf = {
        'num_mel_bins': 128, 
        'target_length': args.target_length, 
        'freqm': 0, 
        'timem': 0, 
        'mixup': 0, 
        'dataset': args.dataset,
        'mode':'train',
        'mean':args.dataset_mean, 
        'std':args.dataset_std, 
        'noise':False}
    
    
    test_loader=get_dataloader(audio_conf,bucket,gcs_prefix,args)
    
    ast_mdl=load_model(args,model_name)
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    ast_mdl.mlp_head[0].register_forward_hook(get_activation('embeddings'))
    
    print('Calculating Embeddings')
    all_names=[]
    all_embeddings=[]
    for inputs,labels,names in tqdm(test_loader):
        logits=ast_mdl(inputs,task=args.task)
        all_names.append(names)
        all_embeddings.append(activation['embeddings'].detach().numpy())
        
    all_names=np.concatenate(all_names)
    all_embeddings=np.concatenate(all_embeddings)
    
    embedding_df=pd.DataFrame(all_embeddings).set_index(all_names)
    embedding_df.columns=[str(s) for s in embedding_df.columns]
    
    return embedding_df



    

    
