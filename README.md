# SSAST for Mayo Data
This is an implementation of the SSAST: Self-supervised audio spectrogram transformer, which is publicly available at
[SSAST github](https://github.com/YuanGongND/ssast). The base model architecture is the same, and original models
can be downloaded from the github to use for finetuning.

## Running requirements
When running with defaults, please download the SSAST-Base-Frame-400 model in the [pretrained model]{https://github.com/YuanGongND/ssast#pretrained-models} section
of the original SSAST github. The code is compatible with other model types, but has not been tested with them.

The environment must include the following packages, all of which can be downloaded with pip:
* opencv-python
* albumentations (has not yet been tested in GCP environment)
* librosa
* torch, torchvision, torchaudio

If running on your local machine and not in a GCP environement, you will also need to install:
* google-cloud
* google-cloud-storage
* google-cloud-bigquery 

If data is stored in GCS, additionally, run 
```gcloud auth application-default login```

```gcloud auth application-defaul set-quota-project PROJECT_NAME```
These allow access to the storage buckets on the local machine.

## Running the SSAST Model
### Data loading
Data is loaded using an AudioDataset class, where you pass a dataframe of the file names (UIDs) along with columns containing label data, a list of the target labels (columns to select from the df), specify audio configuration, method of loading, and initialize transforms on the raw waveform and spectrogram (see [dataloader_mayo.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/dataloader_mayo.py)). This implementation diverges greatly from the original dataloading dataset, especially in that the resulting samples will be a dictionary rather than a tuple. As such, when training/evaluating, you will need to access the fbank and labels as follows: batch['fbank'], batch['targets]. 


To specify audio loading method, you can alter the `bucket` variable and `librosa` variable. As a default, `bucket` is set to None, which will force loading from the local machine. If using GCS, pass a fully initialized bucket. Setting the `librosa` value to 'True' will cause the audio to be loaded using librosa rather than torchaudio. 

The audio configuration parameters should be given as a dictionary (which can be seen in [run_mayo.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/run_mayo.py) and [run_mayo.ipynb]https://github.com/dwiepert/mayo-ssast/blob/main/src/run_mayo.ipynb). Most configuration values are for initializing transforms. The transform will only be initialized if the value is not 0. If you have a further desire to add transforms, see [dataloader_utils.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/utilities/dataloader_utils.py)) and alter [dataloader_mayo.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/dataloader_mayo.py) accordingly. 

The following parameters are accepted:

*Dataset Information*
* `dataset`: a string of the dataset name
* `mode`: either 'train' or 'evaluation'
* `mean`: dataset mean (float)
* `std`: dataset standard deviation (float)
*Audio Transform Information*
* `resample_rate`: an integer value for resampling.
* `reduce`: a boolean indicating whether to reduce audio to monochannel. 
* `clip_length`: integer specifying how many frames the audio should be. 
* `tshift`: Time shifting parameter (between 0 and 1)
* `speed`: Speed tuning parameter (between 0 and 1)
* `gauss_noise`: amount of gaussian noise to add (between 0 and 1)
* `pshift`: pitch shifting parameter (between 0 and 1)
* `pshiftn`: number of steps for pitch shifting
* `gain`: gain parameter (between 0 and 1)
* `stretch`: audio stretching parameter (between 0 and 1)
*Spectrogram Transform Information*
* `num_mel_bins`: number of frequency bins for converting from wav to spectrogram
* `target_length`: target length of resulting spectrogram
* `freqm`: frequency mask paramenter
* `timem`: time mask parameter
* `noise`: add default noise to spectrogram
* `skip_norm`: boolean indicating whether to skip normalization of the spectrogram. 
*Other?*
* `mixup`: parameter for file mixup. This is not currently implemented, so regardless of value, it will not run. 

Outside of the regular audio configurations, you can also set a boolean value for `cdo` (coarse drop out) and `shift` (affine shift). These are remnants of the original SSAST dataloading and not required. Both default to False. 

### Model Classes
One other difference between the original implementation and ours is that we attempted to remove all branching logic in the model initialization so as to make it possible to visualize attention. As such,
we split the orignal `ASTModel class` into `ASTModel_pretrain` and `ASTModel_finetune`. As of now, only the `ASTModel_finetune` class is available. Please see [ast_models.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/models/ast_models.py) for specifics on what arguments it takes and what the default values are. 

### run_mayo.py 
The command line usable, start-to-finish implementation of SSAST is available with [run_mayo.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/run_mayo.py). There is also a notebook implementation: [run_mayo.ipynb](https://github.com/dwiepert/mayo-ssast/blob/main/src/run_mayo.ipynb). This implementation completes fine-tuning and evaluation of a fine-tuned model. It DOES NOT return embeddings. Please see [get_embeddings.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/get_embeddings.py) for this functionality. 

There are many possible arguments to set, including all the parameters associated with audio configuration (see [Data loading]((https://github.com/dwiepert/mayo-ssast#dataloading??)). The main run function describes most of these, and you can alter defaults as required. Run We will list some of the most important.

* `-i`: sets the `prefix` or input directory. Compatible with both local and GCS bucket directories containing audio files, though do not include 'gs://'
* `-d`: sets the `data_split_root` directory. This is a full file path to a directory containing a train.csv and test.csv of file names. This path should include 'gs://' if it is located in a bucket. 
* `-l`: sets the `label_txt` path. This is a full file path to a .txt file contain a list of the target labels for selection (see [labels.txt](https://github.com/dwiepert/mayo-ssast/blob/main/src/labels.txt))
* `-b`: sets the `bucket_name` for GCS loading. Required if loading from cloud.
* `-p`: sets the `project_name` for GCS loading. Required if loading from cloud. 
* `--lib`: specifies whether to load using librosa (True) or torchaudio (False)
* `-o`: sets the `exp_dir`, the directory to save all outputs to. 
* `--task`: string specifying the task to perform. As of now, only compatible with 'ft_cls' and 'ft_avgtok' for fine-tuning. 
* `--batch_size`: set the batch size (default 8)
* `--num_workers`: set number of workers for dataloader (default 0)
* `--epochs`: set number of training epochs
* `--pretrained_mdl_path`: If fine-tuning, this is a required parameter that must be a full file path pointing to a pretrained SSAST model 
* `--freeze`: this is a boolean indicating whether to freeze the model before fine-tuning. It is true as a default
* `--basic`: this is a boolean indicating whether to run a basic training/testing loop or a full training/testing loop with warm up and a lr scheduler. True as a default.
* `--eval_only`: this is a boolean indicating whether to just evaluate an model. You must then specify `--mdl_path`
* `--mdl_path`: this is a path to a final version of a model you wish to evaluate.

Notes: 
- the loss function defaults to binary cross entropy (BCE)
- the optimizer is Adam (for full training loop) or AdamW (for basic training loop) by default
- the code will automatically save the model
- there are options to alter the learning rate and lr scheduler
- there are options for weighted averaging in fine-tuning (`--wa` should be set to True)
- you can alter additional model parameters. 

### New traintest function
We slightly altered the original train/validation functions for fine-tuning. The new version is available at [traintest_mayo.py](https://github.com/dwiepert/mayo-ssast/blob/main/src/traintest_mayo.py)

## Embeddings
Take in a fine-tuned model and get embeddings from the model for an audio file.

## Visualize Attention



