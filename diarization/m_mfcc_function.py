# librosa
import librosa
import librosa.display
import IPython.display

# python_speech_features
import python_speech_features
from python_speech_features import sigproc

# numpy
import numpy
import numpy as np

# plt
import matplotlib.pyplot as plt

# conv
from scipy import signal
from scipy.io import wavfile


# 预处理 分帧
def frames(signal, samplerate=16000, winlen=0.025, winstep=0.01, preemph=0.97,
           winfunc=lambda x: numpy.ones((x,))):
    signal = sigproc.preemphasis(signal, preemph)
    return sigproc.framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)


def read_wav(path):
    # Read wav file and convert to torch tensor
#     sample_rate, audio  = wavfile.read(path)
#     # print(sample_rate)

#     # Resample data if not 16k
#     if (sample_rate != 16000):
#         number_of_samples = round(len(audio) * float(16000) / sample_rate)
#         audio = signal.resample(audio, number_of_samples)

    return librosa.load(path, sr=16000)
#     return audio, 16000
