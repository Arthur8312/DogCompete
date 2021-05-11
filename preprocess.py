import librosa
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
import python_speech_features
import nlpaug.augmenter.audio as ag
DATA_PATH = "./data/"
npy_path = "npy/"

# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


# Handy function to convert wav2mfcc
def wav2mfcc(file_path, max_len=120):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    mfcc_t = python_speech_features.mfcc(wave, samplerate=16000, nfft=2048, nfilt=120)
    mfcc_t = mfcc_t.T
    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc_t.shape[1]):
        pad_width = max_len - mfcc_t.shape[1]
        mfcc_t = np.pad(mfcc_t, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc_t = mfcc_t[:, :max_len]
    
    return mfcc_t

def wav2melspec(file_path, max_len=120):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    melspec = python_speech_features.base.logfbank(wave, samplerate=16000, nfft=2048, nfilt=120)
    melspec = melspec.T
    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > melspec.shape[1]):
        pad_width = max_len - melspec.shape[1]
        melspec = np.pad(melspec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    # Else cutoff the remaining parts
    else:
        melspec = melspec[:, :max_len]
    
    return melspec

def wav2melspec_aug(file_path, max_len=120):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    aug = ag.VtlpAug(sr)
    augmented_data = aug.augment(wave)
    melspec = python_speech_features.base.logfbank(augmented_data, samplerate=16000, nfft=2048, nfilt=120)
    melspec = melspec.T

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > melspec.shape[1]):
        pad_width = max_len - melspec.shape[1]
        melspec = np.pad(melspec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    # Else cutoff the remaining parts
    else:
        melspec = melspec[:, :max_len]
    
    return melspec


def save_data_as_npy_aug(input_path=DATA_PATH, output_path=npy_path, max_len=120):
    labels, _, _ = get_labels(input_path)
    
    for label in labels:
        # Init mfcc vectors
        melspec_list = []

        wavfiles = [input_path + label + '/' + wavfile for wavfile in os.listdir(input_path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
          for i in range(3):
            melspec = wav2melspec_aug(wavfile, max_len=max_len)
            melspec_list.append(melspec)
        np.save(output_path+ label + '.npy', melspec_list)


def save_data_as_npy(input_path=DATA_PATH, output_path=npy_path, max_len=120):
    labels, _, _ = get_labels(input_path)
    
    for label in labels:
        # Init mfcc vectors
        melspec_list = []

        wavfiles = [input_path + label + '/' + wavfile for wavfile in os.listdir(input_path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            melspec = wav2melspec(wavfile, max_len=max_len)
            melspec_list.append(melspec)
        np.save(output_path+ label + '.npy', melspec_list)


def get_train_test(split_ratio=0.9, random_state=42, npy_path=npy_path):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(npy_path + labels[0] + '.npy', allow_pickle=True )
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(npy_path + label + '.npy', allow_pickle=True )
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio),  shuffle=True)


# print(prepare_dataset(DATA_PATH))
if __name__ == "__main__":
  wave, sr = librosa.load('set/bed_g0_1s_16k.wav', mono=True, sr=None)
  # wave = wave[::3]
  mfcc_t = python_speech_features.base.logfbank(wave, samplerate=16000, nfft=2048, nfilt=120)
  # mfcc_test = wav2mfcc('set/bed_g0_1s_16k.wav', 120)
  x = np.load('npy/bed.npy', allow_pickle=True)
  x1 = np.load('npy/cat.npy', allow_pickle=True)