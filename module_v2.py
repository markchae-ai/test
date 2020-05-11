# Modified by J.W.Chae 20-05-11

import os
from pydub import AudioSegment, effects
import librosa
import numpy as np
from python_speech_features import mfcc
#import matplotlib.pyplot as plt
import noisereduce as nr
import ffmpeg_normalize

# 변수를 객체화해서 저장
class dt(): #doctor as dt
    file_name = ''
    
    audio_mp3 = AudioSegment.empty()
    audio_mono = AudioSegment.empty()
    noise_profile_segment = AudioSegment.empty()
        
    audio_array = []
    audio_array_float32 = np.zeros(1)
    noise_profile = np.zeros(1)

    info_user = {}
    info_audio = []  # 2차원, 0 = audio_mono(AudioSegment), 1=audio_array, 2 = audio_array_float32(array), 3 = noise_profile(array)
    info_section = {} # 0 = noise_range(list), 1=voice_range(list), 2= invert_section(list)
    info_developer = {} # phase판단(str)
    audio_process = {}
    all_dic = {}
    audio_info = []
# 목소리와 노이즈를 구분하는 함수
def compute_mfcc(audio_data, sample_rate):
    mfcc_feat = mfcc(audio_data, sample_rate, winlen=0.010, winstep=0.01,
                     numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
                     preemph=0.97, ceplifter=22, appendEnergy=True)
    return mfcc_feat
# 파일을 열어서 변수로 저장하는 함수
def file_open(path_input,data_type):
    audio_orig = AudioSegment.from_file(path_input)
    dt.file_name = f'{os.path.splitext(path_input)[0]}.mp3'

    if data_type == 1:
        if '.mp3' in path_input:
            dt.audio_mp3 = audio_orig
        else:
            audio_orig.export(dt.file_name, format='mp3')
            dt.audio_mp3 = AudioSegment.from_mp3(dt.file_name)
    else:
        audio_orig.export(dt.file_name, format='mp3')
        dt.audio_mp3 = AudioSegment.from_mp3(dt.file_name)
        
    dt.info_user['file_name'] = dt.file_name
    dt.info_user['file_format'] = dt.audio_mp3.frame_rate
    dt.info_user['file_channels'] = dt.audio_mp3.channels
    dt.info_user['file_rms'] = dt.audio_mp3.rms

def get_array(value):
    dt.audio_mp3 = value
    dt.audio_array = value.get_array_of_samples()
    dt.audio_array_float32 = np.array(dt.audio_array).astype(np.float32)
    noise_detect()

def get_array_noise(): #어레이 추출, 채널 분리, 노이즈 디텍팅
    dt.info_audio = []
    audio_orig = effects.normalize(dt.audio_mp3, -1.0)

    if dt.info_user['file_channels'] == 1: #mono파일
        get_array(audio_orig)
        dt.info_audio.append([audio_orig,dt.audio_array,dt.audio_array_float32, dt.noise_profile])
        dt.audio_info.append([dt.audio_array_float32,dt.noise_profile])
    else: #stereo 파일
        ch = audio_orig.split_to_mono()
        for i in ch:
            get_array(i)
            dt.info_audio.append([audio_orig,dt.audio_array,dt.audio_array_float32, dt.noise_profile])
            dt.audio_info.append([dt.audio_array_float32,dt.noise_profile])
    #dt.all_dic['audio_info'] = dt.audio_info
    audio_orig=[]

def noise_detect():
    # 사용 변수
    noise_range = []
    voice_range = []
    noise_rms = 0
    voice_rms = 0
    range_count = 0
    point = 0 
    offset = 0
    npf_check = False
    first_count = True
    # 범위를 찾아주는 함수
    mfcc = compute_mfcc(dt.audio_array_float32/32767, dt.info_user['file_format'])
    cnt = mfcc.shape[0]

    for i in range(cnt):
        if (max(mfcc[i:i + 50, 0]) < -5) and (i > point):
            while (max(mfcc[i:i + offset+50, 0]) < -5):
                offset = offset+1
                if (i + offset + 50) > cnt:
                    break
            starting_point = i * 10
            end_point = (i + offset + 49) * 10
            dt.noise_profile_segment = dt.audio_mp3[starting_point:end_point]
            
            if npf_check == False:
                dt.noise_profile = np.array(dt.noise_profile_segment.get_array_of_samples()).astype(np.float32)                

            npf_check = True
            noise_range.append([starting_point, end_point])
            point = i + offset + 50
            offset = 0
    
    if not noise_range:
        dt.all_dic['noise_sec'] = 'none'
        pass
    elif noise_range[-1][1] > len(dt.audio_mp3):
        noise_range[-1][1] = len(dt.audio_mp3)
        dt.all_dic['noise_sec'] = noise_range[0]

    while range_count < len(noise_range):
        if first_count == True:
            if noise_range[range_count][0] != 0:
                end_value = noise_range[range_count][0] - 1
                voice_range.append([0, end_value])
            first_count = False
        else:
            start_value = noise_range[range_count - 1][1] + 1
            end_value = noise_range[range_count][0] - 1
            voice_range.append([start_value, end_value])            
        range_count += 1
    
    for i in noise_range:
        noise_rms = noise_rms + audio_orig[i[0]:i[1]].rms
    for i in voice_range:
        voice_rms = voice_rms + audio_orig[i[0]:i[1]].rms

    dt.info_section['noise_section'] = noise_range
    dt.info_section['voice_section'] = voice_range
    dt.info_developer['noise_rms'] = noise_rms/len(noise_range)    
    dt.info_developer['voice_rms'] = voice_rms/len(voice_range)
    dt.info_user['noise_rms'] = noise_rms/len(noise_range)
    dt.info_user['volume_noise'] = round((dt.info_developer['noise_rms']/dt.info_developer['voice_rms'])*100,2)
    
    


def stereo_judgement():
    if dt.info_user['file_channels'] ==1: #노이즈 제거로 이동
        dt.info_user['channel_stat'] = 'mono'
        dt.info_user['phase_stat'] = 'good'
        dt.info_developer['phase'] = 'no'
    else:
        left_abs,right_abs = np.fabs(dt.info_audio[0][1]),np.fabs(dt.info_audio[1][1]) #일반 array파일의 절댓값
        left_sum,right_sum = np.sum(left_abs)/(np.sum(left_abs)+np.sum(right_abs)),np.sum(right_abs)/(np.sum(left_abs)+np.sum(right_abs))

        if left_sum < right_sum:
            if right_sum >0.9:  #왼쪽 없음, 노이즈 제거로 이동
                dt.info_user['channel_stat'] = 'left_Less'
                dt.info_user['phase_stat'] = 'good'
                dt.info_developer['phase'] = 'no'
            else: #오른쪽살짝 큰 정상, 위상 판단으로 이동
                dt.info_user['channel_stat'] = 'good' 
                dt.info_developer['phase'] = 'right'

        elif left_sum > right_sum:
            if left_sum >0.9:  #오른쪽 없음, 노이즈 제거로 이동
                dt.info_user['channel_stat'] = 'right_Less'
                dt.info_user['phase_stat'] = 'good'
                dt.info_developer['phase'] = 'no'
            else: #왼쪽 살짝 큰 정상, 위상 판단으로 이동
                dt.info_user['channel_stat'] = 'good' 
                dt.info_developer['phase'] = 'left'
        else: #듀얼모노, 위상 판단으로 이동
            dt.info_user['channel_stat'] = 'good' 
            dt.info_developer['phase'] = 'dual'

def phase_judgement():
    if dt.info_developer['phase'] != 'no':
        #left = dt.info_audio[0][1]
        #right = dt.info_audio[1][1]
        invert_count = []
        invert_section = []
        offset=0
        point=0
        sec =100
        a=0
        while a<len(dt.info_audio[0][1])/sec: #역상 위치파악
            phase_left =np.sum(dt.info_audio[0][1][a*sec:a*sec+sec])
            phase_right =np.sum(dt.info_audio[1][1][a*sec:a*sec+sec])
            phase_sum = phase_left+phase_right
            if abs(phase_sum)*2< (np.fabs(phase_left)+np.fabs(phase_right))/2:
                invert_count.append(1)
            else:
                invert_count.append(0)
            a+=1                
        for i in range(len(invert_count)): #역상여부 결정
            if (np.sum(invert_count[i:i+10])>5)and(i>point):
                offset = i
                startP = i*sec/dt.info_user['file_format']
                while np.sum(invert_count[offset:offset+10])>5:
                    offset+=1
                    if offset+10>len(invert_count):
                        break
                endP = ((offset+10)*sec-1)/dt.info_user['file_format']
                invert_section.append([startP,endP])
                point = offset+10
                offset=0
                break
        if not invert_section:
            dt.info_user['phase_stat'] = 'stereo_perfect'
        else:
            dt.info_user['phase_stat'] = 'invert_error'
        dt.info_section['invert_section'] = invert_section
    else:
        dt.info_user['phase_stat'] = 'good'

def dic():

    dt.all_dic['file_rms'] = dt.info_user['file_rms']
    dt.all_dic['noise_rms'] = dt.info_developer['noise_rms']
    dt.all_dic['noise_aver'] = dt.info_user['noise_rms']
    dt.all_dic['channel_stat'] = dt.info_user['channel_stat']
    dt.all_dic['phase'] = dt.info_developer['phase']
    dt.all_dic['phase_stat'] = dt.info_user['phase_stat']
    dt.all_dic['clarity']= dt.info_user['volume_noise']

    return dt.all_dic
    
    

