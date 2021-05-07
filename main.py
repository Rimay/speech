import wave as wv
from scipy.io import wavfile

import numpy as np
import matplotlib.pyplot as plt

from IPython import embed


wav_file_name = './data/test_a/0AAFP7S8UW.wav'
frame_num, sample_rate = 0, 0

def read_file(file_name):
    global sample_rate, frame_num
    f = wv.open(file_name)
    for item in enumerate(f.getparams()):
        print(item)
    frame_num = f.getparams().nframes
    sample_rate = f.getparams().framerate 


if __name__ == '__main__':
    read_file(wav_file_name)

    a, b = wavfile.read(wav_file_name)

    
    plt.plot(np.arange(0, 1.0 * frame_num / sample_rate, 1.0 / sample_rate), b,'blue')
    plt.xlabel("time (s)")
    plt.show()

    embed()