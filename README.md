# SSAST for Mayo Data
This is an implementation of the SSAST: Self-supervised audio spectrogram transformer, which is publicly available at
[SSAST github](https://github.com/YuanGongND/ssast). The base model architecture is the same, and it is compatible with original pre-trained models that
can be downloaded from the github to use for finetuning. The major changes are that:
1. The original `ASTModel` class was split into `ASTModel_pretrain` and `ASTModel_finetune`, with no branching logic in the forward loop of the finetune model for use in visualizing attention.
2. We added a new classification head with a Dense layer, ReLU activation, LayerNorm, dropout, and a final linear projection layer (this class is defined as `ClassificationHead` in [speech_utils.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/utilities/speech_utils.py))
3. We added options for freezing the base SSAST model in the finetune class.
4. We added an embedding extraction function to the finetune model class.
5. While compatible with old finetuning functions, we also added our training/validation/evaluation loops as well as a loop for embedding extraction. 

The command line usable, start-to-finish implementation for Mayo speech data is available with [run.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/run.py). A notebook tutorial version is also available at [run.ipynb](https://github.com/dwiepert/mayo-ssast/blob/main/src/run.ipynb). This implementation now has options for pre-training a ssast model, fine-tuning a ssast model, evaluating a saved model, or extracting embeddings.

## Known errors
Before installing any packages or attempting to run the code, be aware of the following errors and missing functionality:
1. If you pass any value above 0 for `num_workers`, you will get an error when you attempt to load a batch. Due to lack of GPU, it is unclear if this is only an issue on CPU only machines. 
2. Original implementations not debugged to be compatible with new classification head <- maybe stick the old classification head back in>. Weighted averaging hasn't been debugged.
3. It does not seem like the learning rate warmup is functioning properly, we will be going in to debug this later.

## Running requirements
The environment must include the following packages, all of which can be dowloaded with pip or conda:
* albumentations
* librosa
* torch, torchvision, torchaudio
* tqdm (this is essentially enumerate(dataloader) except it prints out a nice progress bar for you)
* pyarrow

If running on your local machine and not in a GCP environment, you will also need to install:
* google-cloud-storage

The [requirements.txt](https://github.com/dwiepert/mayo-ssast/blob/main/requirements.txt) can be used to set up this environment. 

To access data stored in GCS on your local machine, you will need to additionally run

```gcloud auth application-default login```

```gcloud auth application-defaul set-quota-project PROJECT_NAME```

## Model checkpoints
In order to initialize an SSAST, you must have access to a pretrained model checkpoint. There are a few different checkpoint options which can be found at the [SSAST github](https://github.com/YuanGongND/ssast). The default model used is [SSAST-Base-Frame-400.pth](https://github.com/YuanGongND/ssast#pretrained-models). These model checkpoints can be loaded in two different ways.

1. Use a path to a local directory where the model checkpoint is downloaded. 

2. Use a model checkpoint saved in a GCS bucket. This option can be specified by giving a full file path starting with `gs://BUCKET_NAME/...`. The code will then download this checkpoint locally and reset the checkpoint path to the path it is saved locally. 

## Audio Configuration
Data is loaded using an `AudioDataset` class, where you pass a dataframe of the file names (UIDs) along with columns containing label data, a list of the target labels (columns to select from the df), specify audio configuration, method of loading, and initialize transforms on the raw waveform and spectrogram (see [dataloader_mayo.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/dataloader_mayo.py)). This implementation diverges greatly from the original dataloading dataset, especially in that the resulting samples will be a dictionary rather than a tuple. As such, when training/evaluating, you will need to access the fbank and labels as follows: batch['fbank'], batch['targets]. 

To specify audio loading method, you can alter the `bucket` variable and `librosa` variable. As a default, `bucket` is set to None, which will force loading from the local machine. If using GCS, pass a fully initialized bucket. Setting the `librosa` value to 'True' will cause the audio to be loaded using librosa rather than torchaudio. 

The audio configuration parameters should be given as a dictionary (which can be seen in [run.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/runpy) and [run.ipynb](https://github.com/dwiepert/mayo-ssast/blob/main/src/run.ipynb). Most configuration values are for initializing audio and spectrogram transforms. The transform will only be initialized if the value is not 0. If you have a further desire to add transforms, see [speech_utils.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/utilities/speech_utils.py)) and alter [dataloader_mayo.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/dataloader_mayo.py) accordingly. 

The following parameters are accepted (`--` indicates the command line argument to alter to set it):

*Dataset Information*
* `dataset`: a string of the dataset name. Set with `--dataset`
* `mode`: either 'train' or 'evaluation' (this is handled in the code, the `-m, --mode` argument in the main function is a different setting)
* `mean`: dataset mean (float). Set with `--dataset_mean`
* `std`: dataset standard deviation (float) Set with `--dataset_std`
*Audio Transform Information*
* `resample_rate`: an integer value for resampling. Set with `--resample_rate`
* `reduce`: a boolean indicating whether to reduce audio to monochannel. Set with `--reduce`
* `clip_length`: integer specifying how many frames the audio should be. Set with `--clip_length`
* `tshift`: Time shifting parameter (between 0 and 1). Set with `--tshift`
* `speed`: Speed tuning parameter (between 0 and 1). Set with `--speed`
* `gauss_noise`: amount of gaussian noise to add (between 0 and 1). Set with `--gauss`
* `pshift`: pitch shifting parameter (between 0 and 1). Set with `--pshift`
* `pshiftn`: number of steps for pitch shifting. Set with `--pshiftn`
* `gain`: gain parameter (between 0 and 1).Set with `--gain`
* `stretch`: audio stretching parameter (between 0 and 1). Set with `--stretch`
*Spectrogram Transform Information*
* `num_mel_bins`: number of frequency bins for converting from wav to spectrogram. Set with `--num_mel_bins`
* `target_length`: target length of resulting spectrogram. Set with `--target_length`
* `freqm`: frequency mask paramenter. Set with `--freqm`
* `timem`: time mask parameter. Set with `--timem`
* `noise`: add default noise to spectrogram. Set with `--noise`
* `skip_norm`: boolean indicating whether to skip normalization of the spectrogram. Set with `--skip_norm`
*Other?*
* `mixup`: parameter for file mixup. This is not currently implemented, so regardless of value, it will not run. Set with `--mixup`

Outside of the regular audio configurations, you can also set a boolean value for `cdo` (coarse drop out) and `shift` (affine shift) when initializing the `AudioDataset`. These are remnants of the original SSAST dataloading and not required. Both default to False. 

## Arguments
There are many possible arguments to set, including all the parameters associated with audio configuration. The main run function describes most of these, and you can alter defaults as required. 

### Loading data
* `-i, --prefix`: sets the `prefix` or input directory. Compatible with both local and GCS bucket directories containing audio files, though do not include 'gs://'
* `-s, --study`: optionally set the study. You can either include a full path to the study in the `prefix` arg or specify some parent directory in the `prefix` arg containing more than one study and further specify which study to select here.
* `-d, --data_split_root`: sets the `data_split_root` directory or a full path to a single csv file. For classification, it must be  a directory containing a train.csv and test.csv of file names. If runnning embedding extraction, it should be a csv file. Running evaluation only can accept either a directory or a csv file. This path should include 'gs://' if it is located in a bucket. 
* `-l, --label_txt`: sets the `label_txt` path. This is a full file path to a .txt file contain a list of the target labels for selection (see [labels.txt](https://github.com/dwiepert/mayo-ssast/blob/main/labels.txt))
* `--lib`: : specifies whether to load using librosa (True) or torchaudio (False), default=False

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
* `--pretrained_mdl_path`: specify a pretrained model checkpoint. Default is `SSAST-Base-Fram-400.pth` This is required regardless of whether you include a fine-tuned model path. 
* `--finetuned_mdl_path`: if running eval-only or extraction, you can specify a fine-tuned model to load in. This can either be a local path of a 'gs://' path, that latter of which will trigger the code to download the specified model path to the local machine. 
* `--freeze`: boolean to specify whether to freeze the base model
* `--original_fn`: boolen to specify whether to use original SSAST functions. 
* `--embedding_type`: specify whether embeddings should be extracted from classification head (ft) or base pretrained model (pt)

### Audio transforms
see the audio configurations section for which arguments to set

### Model parameters 
You do not need to worry about `--fstride, --tstride, --fshape, --tshape`. The defaults are fine as is.
* `--task`: string specifying the task to perform. As of now, only compatible with 'ft_cls' and 'ft_avgtok' for fine-tuning. 
* `--model_size`: specify the size of the AST model to initialize. Needs to be compatible with the pretrained model. Default it `base`
* `-pm, --pooling_mode`: specify method of pooling the last hidden layer for embedding extraction. Options are 'mean', 'sum', 'max'.

### Training parameters
* `--batch_size`: set the batch size (default 8)
* `--num_workers`: set number of workers for dataloader (default 0)
* `--learning_rate`: you can manually change the learning rate (default 0.0003)
* `--epochs`: set number of training epochs (default 1)
* `--optim`: specify the training optimizer. Default is `adam`.
* `--loss`: specify the loss function. Can be 'BCE' or 'MSE'. Default is 'BCE'.
* `--scheduler`: specify a lr scheduler. If None, no lr scheduler will be use. The only scheduler option is 'onecycle', which initializes `torch.optim.lr_scheduler.OneCycleLR`
* `--max_lr`: specify the max learning rate for an lr scheduler. Default is 0.01.

### Original fn parameters
Other parameters for the original function may be altered. We do not list them here. Run `python run.py -h` for more information.

### Classification Head parameters
* `--activation`: specify activation function to use for classification head
* `--final_dropout`: specify dropout probability for final dropout layer in classification head
* `--layernorm`: specify whether to include the LayerNorm in classification head

## New traintest function
We slightly altered the original train/validation functions for fine-tuning and pre-training. The new versions are available at [traintest_mayo.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/traintest_mayo.py) for fine-tuning and [traintest_mask_mayo.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/traintest_mask_mayo.py) for pre-training. 

## Embeddings
Embedding extraction is now a function within `ASTModel_finetun` model (see `extract_embeddings(...)` in [ast_models.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/models/ast_models.py)). Notably, this function contains options to extract either the output of the Dense layer from the classification head or the final hidden layer of the base SSAST model by specifying `embedding_type` as either `ft` for 'finetuned' embedding (extracting from classification head) or `pt` for 'pretrained' embedding (extracting from hidden states), on the basis that we generally freeze the model before finetuning. Note that you must indicate a merging strategy for `pt` type embedding extraction to pool the hidden state. This is defaulted to `mean`. It can be set at model initialization or as a parameter in the command line `-pm, --pooling_mode`.

## Visualize Attention
Not yet implemented. 


