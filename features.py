import os
import numpy
import pickle
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
from python_speech_features import mfcc     # Mel Frequency Cepstral Coefficients
from python_speech_features import logfbank # log Mel-filterbank
from python_speech_features import fbank    # Mel-filterbank
from python_speech_features import ssc      # Spectral Subband Centroids


# Get signal features using different feature extraction methods
# https://python-speech-features.readthedocs.io/en/latest/
def get_mfcc_feat(
        file_path, samplerate=16000, winlen=0.1,
        #file_path, samplerate=16000, winlen=0.025,
        winstep=0.01, numcep=13, nfilt=26,
        nfft=4800, lowfreq=0, highfreq=None,
        preemph=0.97, ceplifter=22, appendEnergy=True):
    """Get mfcc feature vector given a signal path.
    @param: file_path – file path of the signal.
    @param: samplerate – the samplerate of the signal we are working with.
    @param: winlen – the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    @param: winstep – the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    @param: numcep – the number of cepstrum to return, default 13
    @param: nfilt – the number of filters in the filterbank, default 26.
    @param: nfft – the FFT size. Default is 2048.
    @param: lowfreq – lowest band edge of mel filters. In Hz, default is 0.
    @param: highfreq – highest band edge of mel filters. In Hz, default is samplerate/2
    @param: preemph – apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    @param: ceplifter – apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    @param: appendEnergy – if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    @return: feature vector (999, 13) (num_frames,  numcep)
    """
    sample_rate, signal = wf.read(file_path)
    feat_vec = mfcc(signal, sample_rate, winlen, winstep, numcep, nfilt, nfft, \
                    lowfreq, highfreq, preemph, ceplifter, appendEnergy)
    return feat_vec


def get_logfbank_feat(
        file_path, samplerate=16000, winlen=0.1,
        winstep=0.01, nfilt=26, nfft=4800,
        lowfreq=0, highfreq=None, preemph=0.97):
    """Get log Mel-filterbank features given a signal path.
    @param: file_path – file path of the signal.
    @param: samplerate – the samplerate of the signal we are working with.
    @param: winlen – the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    @param: winstep – the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    @param: nfilt – the number of filters in the filterbank, default 26.
    @param: nfft – the FFT size. Default is 512.
    @param: lowfreq – lowest band edge of mel filters. In Hz, default is 0.
    @param: highfreq – highest band edge of mel filters. In Hz, default is samplerate/2
    @param: preemph – apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    @return: feature vector (999, 26) (num_frames, nfilt)
    """
    sample_rate, signal = wf.read(file_path)
    feat_vec = logfbank(signal, sample_rate, winlen, winstep, nfilt, \
                        nfft, lowfreq, highfreq, preemph)
    return feat_vec


def get_fbank_feat(
        file_path, samplerate=16000, winlen=0.1,
        winstep=0.01, nfilt=26, nfft=4800,
        lowfreq=0, highfreq=None, preemph=0.97):
    """Get log Mel-filterbank features given a signal path.
    @param: file_path – file path of the signal.
    @param: samplerate – the samplerate of the signal we are working with.
    @param: winlen – the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    @param: winstep – the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    @param: nfilt – the number of filters in the filterbank, default 26.
    @param: nfft – the FFT size. Default is 512.
    @param: lowfreq – lowest band edge of mel filters. In Hz, default is 0.
    @param: highfreq – highest band edge of mel filters. In Hz, default is samplerate/2
    @param: preemph – apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    @return: tuple of feature vector (999, 26) (num_frames, nfilt) and energe vector (999, )
    """
    sample_rate, signal = wf.read(file_path)
    feat_vec, energe_vec = fbank(signal, sample_rate, winlen, winstep, nfilt, \
                                 nfft, lowfreq, highfreq, preemph)
    return feat_vec, energe_vec


def get_ssc_feat(
        file_path, samplerate=16000, winlen=0.025,
        winstep=0.01, nfilt=26, nfft=2048,
        lowfreq=0, highfreq=None, preemph=0.97):
    """Get Spectral Subband Centroid features given a signal path.
    @param: file_path – file path of the signal.
    @param: samplerate – the samplerate of the signal we are working with.
    @param: winlen – the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    @param: winstep – the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    @param: nfilt – the number of filters in the filterbank, default 26.
    @param: nfft – the FFT size. Default is 512.
    @param: lowfreq – lowest band edge of mel filters. In Hz, default is 0.
    @param: highfreq – highest band edge of mel filters. In Hz, default is samplerate/2
    @param: preemph – apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    @return: tuple of feature vector (999, 26) (num_frames, nfilt)
    """
    sample_rate, signal = wf.read(file_path)
    feat_vec = ssc(signal, sample_rate, winlen, winstep, nfilt, \
                     nfft, lowfreq, highfreq, preemph)
    return feat_vec


"""
audio2feat = {}
for file_name in os.listdir("audio"):
    file_path = "audio/{0}".format(file_name)
    audio2feat[file_name] = get_logfbank_feat(file_path)
for file_name in os.listdir("audio-eval-0"):
    file_path = "audio-eval-0/{0}".format(file_name)
    audio2feat[file_name] = get_logfbank_feat(file_path)
for file_name in os.listdir("audio-eval-1"):
    file_path = "audio-eval-1/{0}".format(file_name)
    audio2feat[file_name] = get_logfbank_feat(file_path)
for file_name in os.listdir("audio-eval-2"):
    file_path = "audio-eval-2/{0}".format(file_name)
    audio2feat[file_name] = get_logfbank_feat(file_path)

with open("logfbank.pickle", "wb") as f:
    pickle.dump(audio2feat, f)
audio2feat = {}

for file_name in os.listdir("audio"):
    file_path = "audio/{0}".format(file_name)
    audio2feat[file_name] = get_mfcc_feat(file_path)

for file_name in os.listdir("audio-eval-0"):
    file_path = "audio-eval-0/{0}".format(file_name)
    audio2feat[file_name] = get_mfcc_feat(file_path)

for file_name in os.listdir("audio-eval-1"):
    file_path = "audio-eval-1/{0}".format(file_name)
    audio2feat[file_name] = get_mfcc_feat(file_path)

for file_name in os.listdir("audio-eval-2"):
    file_path = "audio-eval-2/{0}".format(file_name)
    audio2feat[file_name] = get_mfcc_feat(file_path)
with open("mfcc.pickle", "wb") as f:
    pickle.dump(audio2feat, f)
"""

audio2feat = {}
for file_name in os.listdir("audio"):
    file_path = "audio/{0}".format(file_name)
    audio2feat[file_name] = get_ssc_feat(file_path)

for file_name in os.listdir("audio-eval-0"):
    file_path = "audio-eval-0/{0}".format(file_name)
    audio2feat[file_name] = get_ssc_feat(file_path)

for file_name in os.listdir("audio-eval-1"):
    file_path = "audio-eval-1/{0}".format(file_name)
    audio2feat[file_name] = get_ssc_feat(file_path)

for file_name in os.listdir("audio-eval-2"):
    file_path = "audio-eval-2/{0}".format(file_name)
    audio2feat[file_name] = get_ssc_feat(file_path)

with open("ssc_100ms.pickle", "wb") as f:
    pickle.dump(audio2feat, f)