# -*- coding: utf-8 -*-

import os
import pickle
import librosa


save_path = 'D:/dev/task/dog_voice/DogCompete/jarvus/AudioSet/youtube'


''' ---------------------------------
Process 1, Load for others
--------------------------------- '''

folder = ['Alarm', 'Bell', 'Blender', 'Cat', 'Dishes', 'Frying', 'Man',
          'Ring', 'Shaver', 'Toothbrush', 'Vacuum', 'Woman']

with open('dataset_other_desed.pkl', 'rb') as fp:
    dataset = pickle.load(fp)
    
for name in folder:
    for file_name in os.listdir(f'{save_path}/{name}'):
        wav_path = f'{save_path}/{name}/{file_name}'
        s, fs = librosa.load(wav_path, sr=8000)
        dataset.append(s[:40000])
        dataset.append(s[-40000:])
        print(wav_path)
        
print(f"Finished: Total {len(dataset)} clips saved.")

with open('dataset_Other_v1.pkl', "wb") as fp:
    pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL) 


''' ---------------------------------
Process 2, Load for Bark, Howl, Yip
--------------------------------- '''

folder = ['Bark', 'Howl', 'Yip']
for name in folder:
    dataset = []
    for file_name in os.listdir(f'{save_path}/{name}'):
        wav_path = f'{save_path}/{name}/{file_name}'
        s, fs = librosa.load(wav_path, sr=8000)
        dataset.append(s[:40000])
        dataset.append(s[-40000:])
        print(wav_path)
        
    print(f"Finished: {name} Total {len(dataset)} clips saved.")
    
    name = name if name != 'Yip' else 'Cry'
    with open(f'dataset_{name}_v1.pkl', "wb") as fp:
        pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL) 