import os
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio 
import pickle
from tqdm import tqdm

import utils as u
from utils import audio_transform as af

class dataset(Dataset):
    def __init__(self, df_path='./UrbanSound8K/metadata/UrbanSound8K.csv', data_path='./UrbanSound8K/'):
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4
        self.data = []

        self.df = u.load_metadata(df_path)
        self.data_path = data_path
        # for i in tqdm(range(len(df))):
        #     file_path = os.path.join(data_path, df.loc[i, 'relative_path'])
        #     audio = af.open(file_path)
        #     rs_audio = af.resample(audio, self.sr)
        #     rc_audio = af.rechannel(rs_audio, self.channel)
        #     dur_audio = af.pad_trunc(rc_audio, self.duration)
        #     shift_audio = af.time_shift(dur_audio, self.shift_pct)
        #     s_gram = af.spectro_gram(shift_audio, n_mels=64, n_fft=1024, hop_len=None)
        #     aug_s_gram = af.spectro_augment(s_gram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        #     self.data.append([aug_s_gram, df.loc[i, 'classID']])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        file_path = os.path.join(self.data_path, self.df.loc[index, 'relative_path'])
        audio = af.open(file_path)
        rs_audio = af.resample(audio, self.sr)
        rc_audio = af.rechannel(rs_audio, self.channel)
        dur_audio = af.pad_trunc(rc_audio, self.duration)
        shift_audio = af.time_shift(dur_audio, self.shift_pct)
        s_gram = af.spectro_gram(shift_audio, n_mels=64, n_fft=1024, hop_len=None)
        aug_s_gram = af.spectro_augment(s_gram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        
        return [aug_s_gram, self.df.loc[index, 'classID']]