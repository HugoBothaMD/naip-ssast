# -*- coding: utf-8 -*-
'''
Audio Dataset functions for SSAST
Reformated and edited based on code from Yuan Gong (https://github.com/YuanGongND/ssast/tree/main/src/dataloader.py) and David Harwath
some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

Last modified: 05/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: dataloader_mayo.py
'''

#IMPORTS
#third party
import albumentations
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn.functional
from torch.utils.data import Dataset

#local
from utilities import *

        
class AudioDataset(Dataset):
    def __init__(self, annotations_df, target_labels, audio_conf, prefix, bucket=None, librosa=False, cdo=False,shift=False):
        '''
        Dataset that manages audio recordings. 

        :param annotations_df: dataframe containing uid of audio file along with labels
        :type annotations_df: pd.DataFrame
        :param target_labels: list containing the specific target labels to select
        :type target_labels: List[Str]
        :param audio_conf: dictionary containing all information for transforms (audio configuration dict)
        :type audio_conf: dict
        :param prefix: location of files to download - can be either prefix in gcs bucket or input directory
        :type prefix: Str
        :param bucket: Google cloud storage bucket, default=None
        :type bucket: GCS bucket
        :param librosa: boolean indicating loading from librosa as compared to torchaudio
        :type librosa: boolean
        :param cdo: boolean indicating coarse drop out should be used
        :type cdo: boolean
        :param shift: boolean indicating affine shift should be used
        :type shift: boolean

        '''

        #set instance variables
        self.annotations_df = annotations_df
        self.target_labels = target_labels
        self.prefix = prefix
        self.bucket = bucket
        self.lib = librosa #set up using librosa vs. torchaudio for loading

        #AUDIO CONFIGURATION PARAMETERS
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        ### AUDIO TRANSFORMATIONS
        self.resample_rate = self.audio_conf.get('resample_rate') #resample if resample rate != 0 and if resample rate != sample rate
        self.reduce = self.audio_conf.get('reduce') #reduce to monochannel if True
        self.clip_length = self.audio_conf.get('clip_length') #truncate clip to specified length if != 0
        ### AUDIO TRANSFORM INFO - albumentation audio augmentations
        self.tshift = self.audio_conf.get('tshift') #0.9
        self.speed = self.audio_conf.get('speed') #0
        self.gauss = self.audio_conf.get('gauss_noise') #0.8
        self.pshift = self.audio_conf.get('pshift') #0
        self.pshift_n = self.audio_conf.get('pshiftn') #0
        self.gain = self.audio_conf.get('gain') #0.9
        self.stretch =  self.audio_conf.get('stretch') #0

        ### SPECTROGRAM TRANSFORMATIONS
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm') #frequency masking if freqm != 0
        self.timem = self.audio_conf.get('timem') #time masking if timem != 0
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        print('MIXUP NOT CURRENTLY AVAILABLE')
        self.mixup = self.audio_conf.get('mixup') #mixup if mixup != 0
        # print('now using mix-up with rate {:f}'.format(self.mixup))
        
        ## dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        ## skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        ## if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')
        self.target_length = self.audio_conf.get('target_length')

        self.label_num = len(self.target_labels)
        print('number of classes is {:d}'.format(self.label_num))

        if cdo:
            self.tf_co=torch.CoarseDropout(always_apply=True,max_holes=16,min_holes=8)
        else:
            self.tf_co = None
        
        if shift:
            self.tf_shift=torch.Affine(translate_px={'x':(0,0),'y':(0,100)})
        else:
            self.tf_shift = None
        
        self.audio_transform, self.al_transform = self._getaudiotransform() #get audio transforms
        self.spec_transform = self._getspectransform() #get spectrogram transforms
        

    def _getaudiotransform(self):
        '''
        Use audio configuration parameters to initialize classes for audio transformation. 
        Outputs two tranform variables, one for regular audio transformation and one for 
        augmentations using albumentations

        These transformations will always load the audio. 
        :outparam audio_transform: standard transforms
        :outparam al_transform: albumentation augmentation transforms
        '''
        waveform_loader = UidToWaveform(prefix = self.prefix, bucket=self.bucket, lib=self.lib)
        transform_list = [waveform_loader]
        if self.reduce:
            channel_sum = lambda w: torch.sum(w, axis = 0).unsqueeze(0)
            mono_tfm = ToMonophonic(reduce_fn = channel_sum)
            transform_list.append(mono_tfm)
        if self.resample_rate != 0:
            downsample_tfm = Resample(self.resample_rate)
            transform_list.append(downsample_tfm)
        if self.clip_length != 0:
            truncate_tfm = Truncate(length = self.clip_length)
            transform_list.append(truncate_tfm)

        tensor_tfm = ToTensor()
        transform_list.append(tensor_tfm)
        transform = torchvision.transforms.Compose(transform_list)

        #albumentations transforms
        al_transform = []
        if self.tshift != 0:
            tshift = TimeShifting(p=self.tshift)
            al_transform.append(tshift)
        
        if self.speed != 0:
            speed = SpeedTuning(p=self.speed)
            al_transform.append(speed)
        
        if self.gauss != 0:
            gauss = AddGaussianNoise(p=self.gauss)
            al_transform.append(gauss)
        
        if self.pshift != 0:
            pshift = PitchShift(p=self.pshift, n_steps = self.pshiftn)
            al_transform.append(pshift)

        if self.gain != 0:
            gain = Gain(p=self.gain)
            al_transform.append(gain)

        if self.stretch != 0:
            stretch = StretchAudio(p=self.stretch)
            al_transform.append(stretch)

        if al_transform != []:
            al_transform = albumentations.Compose(al_transform)

        # al_transforms = albumentations.Compose([
        # TimeShifting(p=0.9), 
        # #SpeedTuning(p=0.8),
        # AddGaussianNoise(p=0.8),
        # #PitchShift(p=0.5,n_steps=1),
        # Gain(p=0.9),
        # #StretchAudio(p=0.1),

        return transform, al_transform

    def _getspectransform(self):
        '''
        Use audio configuration parameters to initialize classes for spectrogram transformation. 
        Outputs one tranform variable. Will always generate the spectrogram, and has options 
        for frequency/time masking, normalization, and adding noise

        :outparam transform: spectrogram transforms
        '''
        wav2bank = Wav2Fbank(self.target_length, self.melbins, self.tf_co, self.tf_shift, override_wave=False) #override waveform so final sample does not contain the waveform - doing so because the waveforms are not the same shape
        transform_list = [wav2bank]
        if self.freqm != 0:
            freqm = FreqMask(self.freqm)
            transform_list.append(freqm)
        if self.timem != 0: 
            timem = TimeMask(self.timem)
            transform_list.append(timem)
        if not self.skip_norm:
            norm = Normalize(self.norm_mean, self.norm_std)
            transform_list.append(norm)
        if self.noise:
            #TODO:
            noise = Noise()
            transform_list.append(noise)
        transform = torchvision.transforms.Compose(transform_list)
        return transform


    def __getitem__(self, idx):
        '''
        Given an index, load and run transformations then return the sample dictionary

        Will run transformations in this order:
        Standard audio transformations (load audio -> reduce channels -> resample -> clip -> subtract mean) - also convert labels to tensor
        Albumentation transformations (Time shift -> speed tune -> add gauss noise -> pitch shift -> alter gain -> stretch audio)
        Spectrogram transformations (convert to spectrogram -> frequency mask -> time mask -> normalize -> add noise)

        The resulting sample dictionary contains the following info
        'uid': audio identifier
        'waveform': audio (n_channels, n_frames)
        'fbank': spectrogram (target_length, frequency_bins)
        'sample_rate': current sample rate
        'targets': labels for current file as tensor

        '''
    
        #If not doing mix-up
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        uid = self.annotations_df.index[idx] #get uid to load
        targets = self.annotations_df[self.target_labels].iloc[idx].values #get target labels for given uid
        
        sample = {
            'uid' : uid,
            'targets' : targets
        }
        
        sample = self.audio_transform(sample) #load and perform standard transformation
        if self.al_transform != []:
            sample = self.al_transform(sample) #audio augmentations
        
        #TODO: initialize mixup
        mix = Mixup()
        if self.mixup == 0:
            sample= mix(sample, None)

        elif random.random() < self.mixup: 
            mix_sample_idx = random.randint(0, len(self.annotations_df)-1)
            mix_uid = self.annotations_df.index[mix_sample_idx]
            mix_targets = self.annotations_df[self.target_labels].iloc[mix_sample_idx].values
        
            sample2 = {
                'uid': mix_uid,
                'targets': mix_targets
            }
            sample2 = self.audio_transform(sample2) #load and perform standard transformation
            if self.al_transform != []:
                sample2 = self.al_transform(sample2) #audio augmentations

            sample = mix(sample, sample2)
        
        else:
            sample = mix(sample, None)

        sample = self.spec_transform(sample) #convert to spectrogram and perform transformations

        return sample
    

    def __len__(self):
        return len(self.annotations_df)
    