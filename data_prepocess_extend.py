import pickle
import os
import numpy as np
import python_speech_features
import nlpaug.augmenter.audio as ag
import nlpaug.flow as augf
def wav2melspec(wave, sr=8000, max_len=499):
    melspec = python_speech_features.base.logfbank(wave, samplerate=sr, nfft=1024, nfilt=120)
    melspec = melspec.T
    if (max_len > melspec.shape[1]):
        pad_width = max_len - melspec.shape[1]
        melspec = np.pad(melspec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        melspec = melspec[:, :max_len]
    return melspec

pk_list = os.listdir('data_extend/') 

#Bark

# with open('data_extend/'+pk_list[0], 'rb') as fp:
#     dataset = pickle.load(fp)
# bark_extend = [wav2melspec(data) for data in dataset]

# with open('train_npy/Barking.pkl', 'rb') as fw:
#     bark_org  = pickle.load(fw)

# out = bark_extend + bark_org    
# np.save('train_npy/Barking.npy', out)  

#Cry

with open('data_extend/'+pk_list[1], 'rb') as fp:
    dataset = pickle.load(fp)
cry_extend = [wav2melspec(data) for data in dataset]

with open('train_npy/Crying.pkl', 'rb') as fw:
    cry_org  = pickle.load(fw)

cry_out = cry_extend + cry_org    
np.save('train_npy/Crying.npy', cry_out)

#Other
with open('data_extend/'+pk_list[3], 'rb') as fp:
    dataset = pickle.load(fp)
other_extend = [wav2melspec(data) for data in dataset]

with open('train_npy/Other.pkl', 'rb') as fw:
    other_org  = pickle.load(fw)

other_out = other_extend + other_org    
np.save('train_npy/Other.npy', other_out)

#Howl

# with open('data_extend/'+pk_list[2], 'rb') as fp:
#     dataset = pickle.load(fp)

# aug = augf.Sequential([ag.VtlpAug(8000, zone=(0,1), coverage=1)])
# howl_temp = []
# for wave in dataset:
#     aug_datas = aug.augment(wave, 2)
#     howl_temp.extend(aug_datas)
#     howl_temp.append(wave)
# howl_extend = [wav2melspec(data) for data in howl_temp]

# with open('train_npy/Howling.pkl', 'rb') as fw:
#     howl_org  = pickle.load(fw)


# howl_list = howl_extend + howl_org    


# np.save('train_npy/Howling.npy', howl_list)