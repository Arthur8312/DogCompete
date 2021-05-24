# -*- coding: utf-8 -*-

'''
Dataset introduction:
    https://zenodo.org/record/4560759#.YKoLMKgzZPY
Dowbload files:
    https://zenodo.org/record/4560759/files/DESED_public_eval.tar.gz?download=1

Dataset introduction:
    https://zenodo.org/record/1213793#.YKoAiqgzZPY
Dowbload files:
    https://zenodo.org/record/1213793/files/Isolated%20urban%20sound%20database.zip?download=1

Dataset introduction:
    https://zenodo.org/record/2535878#.YKoKVagzZPY
Dowbload files:
    https://zenodo.org/record/2535878/files/NIGENS.zip?download=1

'''

import pickle
import librosa
import pandas as pd

''' ---------------------------------
Step 1, Read csv file
--------------------------------- '''

wav_folder_path = 'DESED_public_eval/dataset/audio/eval/public'
label_path = 'DESED_public_eval/dataset/metadata/eval/public.csv'
df = pd.read_csv(label_path, sep='	')


''' ---------------------------------
Step 2, Organize audio label
--------------------------------- '''

data_dict = {}

for idx in range(len(df)):
    
    row = df.iloc[idx]
    file_name = row.filename
    label = row.event_label
    
    if file_name not in data_dict:
        data_dict[file_name] = []
    if label not in data_dict[file_name]:
        data_dict[file_name].append(label)
        
''' ---------------------------------
Step 3, Extract audio except from dog
--------------------------------- '''

dataset = []

for file_name in data_dict:
    if 'Dog' not in data_dict[file_name]:

        wav_path = f'{wav_folder_path}/{file_name}'
        s, fs = librosa.load(wav_path, sr=8000)

        dataset.append(s[:40000])
        dataset.append(s[-40000:])
        print(wav_path)

print(f"Finished: Total {len(dataset)} clips saved.")

''' ---------------------------------
Step 4, Save into pickle dict
--------------------------------- '''

with open('dataset_other_desed.pkl', "wb") as fp:
    pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL) 













