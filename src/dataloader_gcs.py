# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader.py.py

# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch
import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random

import io 

import cv2 
import random

#sound
import librosa

#albumentations core
from albumentations.core.transforms_interface import DualTransform, BasicTransform

class AudioTransform(BasicTransform):
    """ Transform for audio task. This is the main class where we override the targets and update params function for our need"""

    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params

class TimeShifting(AudioTransform):
    """ Do time shifting of audio """
    def __init__(self, always_apply=False, p=0.5):
        super(TimeShifting, self).__init__(always_apply, p)
        
    def apply(self,data,**params):
        '''
        data : ndarray of audio timeseries
        '''        
        start_ = int(np.random.uniform(-80000,80000))
        if start_ >= 0:
            audio_time_shift = np.r_[data[start_:], np.random.uniform(-0.001,0.001, start_)]
        else:
            audio_time_shift = np.r_[np.random.uniform(-0.001,0.001, -start_), data[:start_]]
        
        return audio_time_shift
    
class SpeedTuning(AudioTransform):
    """ Do speed Tuning of audio """
    def __init__(self, always_apply=False, p=0.5,speed_rate = None):
        '''
        Give Rate between (0.5,1.5) for best results
        '''
        super(SpeedTuning, self).__init__(always_apply, p)
        
        if speed_rate:
            self.speed_rate = speed_rate
        else:
            self.speed_rate = np.random.uniform(0.6,1.3)
        
    def apply(self,data,**params):
        '''
        data : ndarray of audio timeseries
        '''        
        audio_speed_tune = cv2.resize(data, (1, int(len(data) * self.speed_rate))).squeeze()
        if len(audio_speed_tune) < len(data) :
            pad_len = len(data) - len(audio_speed_tune)
            audio_speed_tune = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
                                   audio_speed_tune,
                                   np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
        else: 
            cut_len = len(audio_speed_tune) - len(data)
            audio_speed_tune = audio_speed_tune[int(cut_len/2):int(cut_len/2)+len(data)]
        
        return audio_speed_tune
    
class StretchAudio(AudioTransform):
    """ Do stretching of audio file"""
    def __init__(self, always_apply=False, p=0.5 , rate = None):
        super(StretchAudio, self).__init__(always_apply, p)
        
        if rate:
            self.rate = rate
        else:
            self.rate = np.random.uniform(0.5,1.5)
        
    def apply(self,data,**params):
        '''
        data : ndarray of audio timeseries
        '''        
        input_length = len(data)
        
        data = librosa.effects.time_stretch(data,self.rate)
        
        if len(data)>input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

        return data
    
class PitchShift(AudioTransform):
    """ Do time shifting of audio """
    def __init__(self, always_apply=False, p=0.5 , n_steps=None):
        super(PitchShift, self).__init__(always_apply, p)
        '''
        nsteps here is equal to number of semitones
        '''
        
        self.n_steps = n_steps
        
    def apply(self,data,**params):
        '''
        data : ndarray of audio timeseries
        '''        
        return librosa.effects.pitch_shift(data,sr=22050,n_steps=self.n_steps)
    
    
class AddGaussianNoise(AudioTransform):
    """ Do time shifting of audio """
    def __init__(self, always_apply=False, p=0.5):
        super(AddGaussianNoise, self).__init__(always_apply, p)
        
        
    def apply(self,data,**params):
        '''
        data : ndarray of audio timeseries
        ''' 
        noise = np.random.randn(len(data))
        data_wn = data + 0.005*noise
        return data_wn
    
class Gain(AudioTransform):
    """
    Multiply the audio by a random amplitude factor to reduce or increase the volume. This
    technique can help a model become somewhat invariant to the overall gain of the input audio.
    """

    def __init__(self, min_gain_in_db=-12, max_gain_in_db=12, always_apply=False,p=0.5):
        super(Gain,self).__init__(always_apply,p)
        assert min_gain_in_db <= max_gain_in_db
        self.min_gain_in_db = min_gain_in_db
        self.max_gain_in_db = max_gain_in_db


    def apply(self, data, **args):
        amplitude_ratio = 10**(random.uniform(self.min_gain_in_db, self.max_gain_in_db)/20)
        return data * amplitude_ratio
    
class CutOut(AudioTransform):
    def __init__(self, always_apply=False, p=0.5 ):
        super(CutOut, self).__init__(always_apply, p)
        
    def apply(self,data,**params):
        '''
        data : ndarray of audio timeseries
        '''
        start_ = np.random.randint(0,len(data))
        end_ = np.random.randint(start_,len(data))
        
        data[start_:end_] = 0
        
        return data
    
import albumentations

def get_train_transforms():
    return albumentations.Compose([
        TimeShifting(p=0.9), 
        #SpeedTuning(p=0.8),
        AddGaussianNoise(p=0.8),
        #PitchShift(p=0.5,n_steps=1),
        Gain(p=0.9),
        #StretchAudio(p=0.1),
    ])

def load_waveform_from_gcs(bucket, gcs_prefix, uid, extension = 'mp3'):
    
    try:
        gcs_waveform_path = f'{gcs_prefix}/{uid}/waveform.{extension}'
        blob = bucket.blob(gcs_waveform_path)
        wave_string = blob.download_as_string()
        wave_bytes = io.BytesIO(wave_string)
    except:
        gcs_waveform_path = f'{gcs_prefix}/{uid}/waveform.wav'
        extension='wav'
        blob = bucket.blob(gcs_waveform_path)
        wave_string = blob.download_as_string()
        wave_bytes = io.BytesIO(wave_string)
    gcs_metadata_path = f'{gcs_prefix}/{uid}/metadata.json'
    
    waveform, _ = torchaudio.load(wave_bytes, format = extension)
    
    metadata_blob = bucket.blob(gcs_metadata_path)
    metadata = json.loads(metadata_blob.download_as_string())
    
    return waveform, metadata

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudioDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, bucket=None, gcs_prefix=None,label_csv=None,cdo=False,shift=False,ct=False,wt=False,resample=False,
                resample_rate=44100):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.bucket=bucket
        self.gcs_prefix=gcs_prefix
        
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))
        self.reduce_fn = lambda w: torch.sum(w, axis = 0).unsqueeze(0)
        
        self.cdo=cdo
        #self.tf_co=CoarseDropout(always_apply=True,max_holes=16,min_holes=8)
        
        self.shift=shift
        #self.tf_shift=Affine(translate_px={'x':(0,0),'y':(0,100)})
        
        self.ct=ct
        self.resample=resample
        self.resample_rate=resample_rate
        """
        self.compose_transform = ComposeTransform([
            RandomClip(sample_rate=44100, clip_length=120000),
            RandomSpeedChange(44100)])
        """
        
        self.wt=wt
        self.wave_transforms=get_train_transforms()

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            
            waveform, metadata = load_waveform_from_gcs(self.bucket,self.gcs_prefix,filename)
            sr = metadata['sample_rate_hz']
            waveform = self.reduce_fn(waveform)
            
            if self.resample:
                
                sr=self.resample_rate
                waveform = torchaudio.transforms.Resample(sr, self.resample_rate)(waveform)
                
            waveform = waveform - waveform.mean()
            
        
        # mixup
        else:
            waveform1, metadata = load_waveform_from_gcs(self.bucket,self.gcs_prefix,filename)
            waveform1 = self.reduce_fn(waveform1)
        
            sr = metadata['sample_rate_hz']
            waveform2,_ = load_waveform_from_gcs(self.bucket,self.gcs_prefix,filename2)
            waveform2 = self.reduce_fn(waveform2)
            
            if self.resample:
                sr=self.resample_rate
                waveform1= torchaudio.transforms.Resample(sr, self.resample_rate)(waveform1)
                waveform2= torchaudio.transforms.Resample(sr, self.resample_rate)(waveform2)
        
        
            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
            
            if self.wt:
                waveform=torch.tensor(self.wave_transforms(data=np.array(waveform[0])['data'],rate=sr))
            if self.ct:
                waveform=self.compose_transform(waveform)
                
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        
        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
            
        if self.cdo:
            fbank=torch.FloatTensor((self.tf_co(image=fbank.numpy()))['image'])
        
        if self.shift:
            fbank=torch.FloatTensor((self.tf_shift(image=fbank.numpy()))['image'])
        
        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup:
            datum = self.data[index]
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]
            # get the mixed fbank
            fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            """
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += (1.0-mix_lambda)
            label_indices = torch.FloatTensor(label_indices)
            """
            for label_str in datum['labels']:
                print(label_str)
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels']:
                label_indices[int(self.index_dict[label_str])] += (1.0-mix_lambda)
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            datum = self.data[index]
            label_indices = np.zeros(self.label_num)
            fbank, mix_lambda = self._wav2fbank(datum['wav'])
            
            label_indices=torch.FloatTensor(datum['labels'])
            
            #for label_str in datum['labels'].split(','):
            #    label_indices[int(self.index_dict[label_str])] = 1.0
            #label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        # this is just to satisfy new torchaudio version.
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        # squeeze back
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, label_indices, datum['wav']

    def __len__(self):
        return len(self.data)