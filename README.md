# SSAST for Mayo Data
This is an implementation of the SSAST: Self-supervised audio spectrogram transformer, which is publicly available at
[SSAST github](https://github.com/YuanGongND/ssast). The base model architecture is the same, and it is compatible with original pre-trained models that
can be downloaded from the github to use for finetuning. The major changes are that:
1. The original `ASTModel` class was split into `ASTModel_pretrain` and `ASTModel_finetune`, with no branching logic in the forward loop of the finetune model for use in visualizing attention.
2. We added a new classification head with a Dense layer, ReLU activation, LayerNorm, dropout, and a final linear projection layer (this class is defined as `ClassificationHead` in [speech_utils.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/utilities/speech_utils.py))
3. We added options for freezing the base SSAST model in the finetune class.
4. We added an embedding extraction function to the finetune model class.
5. We added our training/validation/evaluation loops as well as a loop for embedding extraction. It is no longer compatible with old training/evaluation functions.

The command line usable, start-to-finish implementation for Mayo speech data is available with [run.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/run.py). A notebook tutorial version is also available at [run.ipynb](https://github.com/dwiepert/mayo-ssast/blob/main/src/run.ipynb). This implementation now has options for pre-training a ssast model, fine-tuning a ssast model, evaluating a saved model, or extracting embeddings.

## Known errors
Before installing any packages or attempting to run the code, be aware of the following errors and missing functionality:
1. If you pass any value above 0 for `num_workers` on a local machine, you might get an error when you attempt to load a batch. Due to lack of GPU, it is unclear if this is only an issue on CPU only machines. 


## Running requirements
The environment must include the following packages, all of which can be dowloaded with pip or conda:
* albumentations
* librosa
* torch, torchvision, torchaudio
* tqdm (this is essentially enumerate(dataloader) except it prints out a nice progress bar for you)
* pyarrow
* timm == '0.4.5'

If running on your local machine and not in a GCP environment, you will also need to install:
* google-cloud-storage

The [requirements.txt](https://github.com/dwiepert/mayo-ssast/blob/main/requirements.txt) can be used to set up this environment. 

To access data stored in GCS on your local machine, you will need to additionally run

```gcloud auth application-default login```

```gcloud auth application-defaul set-quota-project PROJECT_NAME```

Please note that if using GCS, the model expects arguments like model paths or directories to start with `gs://BUCKET_NAME/...` with the exception of defining an output cloud directory which should just be the prefix to save within a bucket. 

## Model checkpoints
In order to initialize an SSAST, you must have access to a pretrained model checkpoint. There are a few different checkpoint options which can be found at the [SSAST github](https://github.com/YuanGongND/ssast). The default model used is [SSAST-Base-Frame-400.pth](https://github.com/YuanGongND/ssast#pretrained-models). These model checkpoints can be loaded in two different ways.

1. Use a path to a local directory where the model checkpoint is downloaded. 

2. Use a model checkpoint saved in a GCS bucket. This option can be specified by giving a full file path starting with `gs://BUCKET_NAME/...`. The code will then download this checkpoint locally and reset the checkpoint path to the path it is saved locally. 

## Data structure
This code will only function with the following data structure.

SPEECH DATA DIR

    |

    -- UID 

        |

        -- waveform.EXT (extension can be any audio file extension)

        -- metadata.json (containing the key 'encoding' (with the extension in capital letters, i.e. mp3 as MP3), also containing the key 'sample_rate_hz' with the full sample rate)

and for the data splits

DATA SPLIT DIR

    |

    -- train.csv

    -- test.csv
    
## Audio Configuration
Data is loaded using an `AudioDataset` class, where you pass a dataframe of the file names (UIDs) along with columns containing label data, a list of the target labels (columns to select from the df), specify audio configuration, method of loading, and initialize transforms on the raw waveform and spectrogram (see [dataloader.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/dataloader.py)). This implementation diverges greatly from the original dataloading dataset, especially in that the resulting samples will be a dictionary rather than a tuple. As such, when training/evaluating, you will need to access the fbank and labels as follows: batch['fbank'], batch['targets]. 

To specify audio loading method, you can alter the `bucket` variable and `librosa` variable. As a default, `bucket` is set to None, which will force loading from the local machine. If using GCS, pass a fully initialized bucket. Setting the `librosa` value to 'True' will cause the audio to be loaded using librosa rather than torchaudio. 

The audio configuration parameters should be given as a dictionary (which can be seen in [run.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/runpy) and [run.ipynb](https://github.com/dwiepert/mayo-ssast/blob/main/src/run.ipynb). Most configuration values are for initializing audio and spectrogram transforms. The transform will only be initialized if the value is not 0. If you have a further desire to add transforms, see [speech_utils.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/utilities/speech_utils.py)) and alter [dataloader.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/dataloader.py) accordingly. 

The following parameters are accepted (`--` indicates the command line argument to alter to set it):

*Dataset Information*
* `mean`: dataset mean (float). Set with `--dataset_mean`
* `std`: dataset standard deviation (float) Set with `--dataset_std`
*Audio Transform Information*
* `resample_rate`: an integer value for resampling. Set with `--resample_rate`
* `reduce`: a boolean indicating whether to reduce audio to monochannel. Set with `--reduce`
* `clip_length`: float specifying how many seconds the audio should be. Will work with the 'sample_rate' of the audio to get # of frames. Set with `--clip_length`
* `tshift`: Time shifting parameter (between 0 and 1). Set with `--tshift`
* `speed`: Speed tuning parameter (between 0 and 1). Set with `--speed`
* `gauss_noise`: amount of gaussian noise to add (between 0 and 1). Set with `--gauss`
* `pshift`: pitch shifting parameter (between 0 and 1). Set with `--pshift`
* `pshiftn`: number of steps for pitch shifting. Set with `--pshiftn`
* `gain`: gain parameter (between 0 and 1).Set with `--gain`
* `stretch`: audio stretching parameter (between 0 and 1). Set with `--stretch`
* `mixup`: parameter for file mixup (between 0 and 1). Set with `--mixup`
*Spectrogram Transform Information*
* `num_mel_bins`: number of frequency bins for converting from wav to spectrogram. Set with `--num_mel_bins`
* `target_length`: target length of resulting spectrogram. Set with `--target_length`
* `freqm`: frequency mask paramenter. Set with `--freqm`
* `timem`: time mask parameter. Set with `--timem`
* `noise`: add default noise to spectrogram. Set with `--noise`

Outside of the regular audio configurations, you can also set a boolean value for `cdo` (coarse drop out) and `shift` (affine shift) when initializing the `AudioDataset`. These are remnants of the original SSAST dataloading and not required. Both default to False. 

## Arguments
There are many possible arguments to set, including all the parameters associated with audio configuration. The main run function describes most of these, and you can alter defaults as required. 

### Loading data
* `-i, --prefix`: sets the `prefix` or input directory. Compatible with both local and GCS bucket directories containing audio files, though do not include 'gs://'
* `-s, --study`: optionally set the study. You can either include a full path to the study in the `prefix` arg or specify some parent directory in the `prefix` arg containing more than one study and further specify which study to select here.
* `-d, --data_split_root`: sets the `data_split_root` directory or a full path to a single csv file. For classification, it must be  a directory containing a train.csv and test.csv of file names. If runnning embedding extraction, it should be a csv file. Running evaluation only can accept either a directory or a csv file. This path should include 'gs://' if it is located in a bucket. 
* `-l, --label_txt`: sets the `label_txt` path. This is a full file path to a .txt file contain a list of the target labels for selection (see [labels.txt](https://github.com/dwiepert/mayo-ssast/blob/main/labels.txt))
* `--lib`: : specifies whether to load using librosa (True) or torchaudio (False), default=False
* `--pretrained_mdl_path`: specify a pretrained model checkpoint. Default is `SSAST-Base-Frame-400.pth` This is required regardless of whether you include a fine-tuned model path. 
* `--finetuned_mdl_path`: if running eval-only or extraction, you can specify a fine-tuned model to load in. This can either be a local path of a 'gs://' path, that latter of which will trigger the code to download the specified model path to the local machine. 

### Google cloud storage
* `-b, --bucket_name`: sets the `bucket_name` for GCS loading. Required if loading from cloud.
* `-p, --project_name`: sets the `project_name` for GCS loading. Required if loading from cloud. 
* `--cloud`: this specifies whether to save everything to GCS bucket. It is set as True as default.

### Saving data
* `--dataset`: Specify the name of the dataset you are using. When saving, the dataset arg is used to set file names. If you do not specify, it will assume the lowest directory from data_split_root. Default is None. 
* `-o, --exp_dir`: sets the `exp_dir`, the LOCAL directory to save all outputs to. 
* `--cloud_dir`: if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket. Do not include the bucket_name or 'gs://' in this path.

### Run mode
* `-m, --mode`: Specify the mode you are running, i.e., whether to run fine-tuning for classification ('finetune'), evaluation only ('eval-only'), or embedding extraction ('extraction'). Default is 'finetune'.
* `--task`: Specify pretraining or fine-tuning task. Choices are 'pretrain_mpc', 'pretrain_mpg', 'pretrain_joint', 'ft_cls', 'ft_avgtok'. 
* `--freeze`: boolean to specify whether to freeze the base model
* `--weighted`: boolean to specify whether to train the weight sum of layers
* `--layer`: Specify which model layer (hidden state) output to use. Default is -1 which is the final layer. 
* `--embedding_type`: specify whether embeddings should be extracted from classification head (ft) or base pretrained model (pt) or weighted model (wt)

### Audio transforms
see the audio configurations section for which arguments to set

### Model parameters 
You do not need to worry about `--fstride, --tstride, --fshape, --tshape`. The defaults are fine as is.
* `--model_size`: specify the size of the AST model to initialize. Needs to be compatible with the pretrained model. Default it `base`

### Training parameters
* `--batch_size`: set the batch size (default 8)
* `--num_workers`: set number of workers for dataloader (default 0)
* `--learning_rate`: you can manually change the learning rate (default 0.0003)
* `--epochs`: set number of training epochs (default 1)
* `--optim`: specify the training optimizer. Default is `adam`.
* `--weight_decay`: specify weight decay for AdamW optimizer
* `--loss`: specify the loss function. Can be 'BCE' or 'MSE'. Default is 'BCE'.
* `--scheduler`: specify a lr scheduler. If None, no lr scheduler will be use. The only scheduler option is 'onecycle', which initializes `torch.optim.lr_scheduler.OneCycleLR`
* `--max_lr`: specify the max learning rate for an lr scheduler. Default is 0.01.

### Pretraining parameters
* `--mask_patch`: how many patches to mask (used only for ssl pretraining)
* `--cluster_factor`: mask clutering factor.
Other parameters for the original function may be altered. We do not list them here. Run `python run.py -h` for more information.

### Classification Head parameters
* `--activation`: specify activation function to use for classification head
* `--final_dropout`: specify dropout probability for final dropout layer in classification head
* `--layernorm`: specify whether to include the LayerNorm in classification head

For more information on arguments, you can also run `python run.py -h`. 

## Functionality
This implementation contains many functionality options as listed below:

### 1. Pretraining
You can pretrain an SSAST model from scratch using the `ASTModel_pretrain` class in [ast_models.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/models/ast_models.py)and the `pretrain(...)` function in [loops.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/loops.py). 

This mode is triggered by setting `-m, --mode` to 'train' and also specifying which pretraining task to use with `--task`. The options are 'pretrain_mpc', 'pretrain_mpg', or 'pretrain_joint' which uses both previous tasks.

Additionally, there are data augmentation transforms available for pretraining, such as time shift, speed tuning, adding noise, pitch shift, gain, stretching audio, and audio mixup. 

This implementation currently can not continue pretraining from an already pretrained model checkpoint. 

### 2. Finetuning
You can finetune SSAST for classifying speech features using the `ASTModel_finetune` class in [ast_models.py]((https://github.com/dwiepert/mayo-ssast/blob/main/src/models/ast_models.py) and the `finetune(...)` function in [loops.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/loops.py). 

This mode is triggered by setting `-m, --mode` to 'train' and also specifying which finetuning task to use with `--task`. The options are 'ft_cls' or 'ft_avgtok'. See `_cls(x)` and `_avgtok(x)` in `ASTModel_finetune` for more information on how merging is done. 

There are a few different parameters to consider. Firstly, the classification head can be altered to use a different amount of dropout and to include/exclude layernorm. See `ClassificationHead` class in [speech_utils.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/utilities/speech_utils.py) for more information. 

Default run mode will also freeze the base AST model and only finetune the classification head. This can be altered with `--freeze`. 

We also include the option to use a different hidden state output as the input to the classification head. This can be specified with `--layer` and must be an integer between 0 and `model.n_states` (or -1 to get the final layer). This works in the `ASTModel_finetune` class by getting a list of hidden states and indexing using the `layer` parameter. The hidden states output will always have the following trait: the last hidden state is run through a normalization layer such that the second to last index is the last hidden state prior to this normalization and the last index is the final output. That is `[output 1, ..... output12, norm(output12)]`. 

Additionally, there are data augmentation transforms available for finetuning, such as time shift, speed tuning, adding noise, pitch shift, gain, stretching audio, and audio mixup. 

Finally, we added functionality to train an additional parameter to learn weights for the contribution of each hidden state (excluding the final output, i.e. hidden_states[:-1]) to classification. The weights can be accessed with `ASTModel_finetune.weightsum`. This mode is triggered by setting `--weighted` to True. If initializing a model outside of the run function, it is still triggered with an argument called `weighted`. 

### 3. Evaluation only
If you have a finetuned model and want to evaluate it on a new data set, you can do so by setting `-m, --mode` to 'eval'. You must then also specify a `--finetuned_mdl_path` to load in. 

It is expected that there is an `args.pkl` file in the same directory as the finetuned model to indicate which arguments were used to initialize the finetuned model. This implementation will load the arguments and initialize/load the finetuned model with these arguments. If no such file exists, it will use the arguments from the current run, which could be incompatible if you are not careful. 


### 4. Embedding extraction.
We implemented multiple embedding extraction methods for use with the SSAST model. The implementation is a function within `ASTModel_finetune` called `extract_embedding(x, embedding_type, layer, task)`, which is called on batches instead of the forward function. 

Embedding extraction is triggered by setting `-m, --mode` to 'extraction'. 

You must also consider where you want the embeddings to be extracted from. The options are as follows:
1. From the output of a hidden state? Set `embedding_type` to 'pt'. Can further set an exact hidden state with the `layer` argument. By default, it will use the layer specified at the time of model initialization. The model default is to give the last hidden state run through a normalization layer - ind 13, so the embedding is this output merged to be of size (batch size, embedding_dim). It will also automatically use the merging strategy defined by the task set at the time of model initialization, but this can be changed at the time of embedding extraction by redefining `task` with either 'ft_cls' or 'ft_avgtok'.
2. After weighting the hidden states? Set `embedding_type` to 'wt'. This version requires that the model was initially finetuned with  `weighted` set to True.
3. From a layer in the classification head that has been finetuned? Set `embedding_type` to 'ft'. This version requires no further specification and will always return the output from the first dense layer in the classification head, prior to any activation function or normalization. 

Brief note on target labels:
Embedding extraction is the only mode where target labels are not required. You can give None or an empty list or np.array and it will still function and extract embeddings.

## Visualize Attention
Not yet implemented. 


