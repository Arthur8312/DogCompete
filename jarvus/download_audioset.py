# -*- coding: utf-8 -*-

'''
https://research.google.com/audioset/ontology/dog_1.html
'''

import os
import threading

import pafy
import pandas as pd
import soundfile as sf 


''' ---------------------------------
Step 1, Read csv file
--------------------------------- '''

save_path = 'D:/dev/task/dog_voice/DogCompete/jarvus/AudioSet/youtube'
csv_path = 'AudioSet/balanced_train_segments.csv'
#csv_path = 'AudioSet/eval_segments.csv'
#csv_path = 'AudioSet/unbalanced_train_segments.csv'

df = pd.read_csv(csv_path, sep=', ')

# Unbalanced next from 1584905
select = {}
#select['Bark'] = '/m/05tny_'
#select['Yip'] = '/m/07r_k2n'
#select['Howl'] = '/m/07qf0zm'

select['Blender'] = '/m/02pjr4'
select['Cat'] = '/m/01yrx'
select['Dishes'] = '/m/04brg2'
select['Toothbrush'] = '/m/04fgwm'
select['Shaver'] = '/m/02g901'
select['Vacuum'] = '/m/0d31p'
select['Alarm'] = '/m/07pp_mv'
select['Bell'] = '/m/0395lw'
select['Ring'] = '/m/07pp8cl'
select['Frying'] = '/m/0dxrf'
select['Man'] = '/m/05zppz'
select['Woman'] = '/m/02zsn'

for key in select:
    if not os.path.exists(f'{save_path}/{key}'):
        os.makedirs(f'{save_path}/{key}')

''' ---------------------------------
Step 2, Download from youtube
--------------------------------- '''

def convert_to_wav_8k(ori_path, output_path):
    command = f'ffmpeg -y -i "{ori_path}" -f wav -ac 1 -ar 8000 -vn "{output_path}" -loglevel quiet'
    os.system(command)
    os.remove(ori_path)
    
def download_youtube(label, audio_id, start_time, end_time):

    file_name = f'{audio_id}_{start_time}_{end_time}'
    print(f'{label}/{file_name}.wav')
    try:
        url = f'https://www.youtube.com/watch?v={audio_id}'
        video = pafy.new(url)
        bestaudio = video.getbestaudio()
        file_path = f'{save_path}/{file_name}.{bestaudio.extension}'
        bestaudio.download(filepath=file_path)
        file_path_wav = f'{save_path}/{label}/{file_name}.wav'
        
        convert_to_wav_8k(file_path, file_path_wav)
    
        data, fs = sf.read(file_path_wav)
        start_point = int(fs*start_time)
        end_point   = int(fs*end_time)
        
        sf.write(file_path_wav, data[start_point:end_point], fs)
    
    except Exception as e:
        print(e)


job_num = 10

def thread_job(num):
    
    job_per = int(len(df)/job_num)
    for idx in range(num*job_per, (num+1)*job_per):
        
        row = df.iloc[idx]
        audio_id = row.youtube
        start_time = int(row.start_seconds)
        end_time = int(row.end_seconds)
        labels = row.positive_labels[1:-1]
        label_list = labels.split(',')
        
        select_tmp = [key for key in select if select[key] in label_list]
        if len(select_tmp) == 1:
            download_youtube(select_tmp[0], audio_id, start_time, end_time)


threads = []
for i in range(job_num):
  threads.append(threading.Thread(target=thread_job, args = (i,)))
  threads[i].start()

for i in range(job_num):
  threads[i].join()