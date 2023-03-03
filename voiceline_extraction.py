import pathlib

import playsound
import srt
import ass
import os
import shutil
import ffmpeg
import subprocess
from datetime import timedelta
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.playback import play
import simpleaudio

def extract_clip_srt(video_path,subtitle:srt.Subtitle, output_dir=None, ext:str=None,start_delta=0.5,end_delta=0.5):
    '''
    Extract audio clips according to timestamps from srt files
    extract a single clip according to the subtitle
    return result's filename and text
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    f = os.path.splitext(video_path)
    fname = f[0]
    if ext is None:
        ext = f[1]
    if output_dir is None:
        output_fname = fname+ext
    else:
        output_fname = output_dir + '/'+str(subtitle.index)+ ext
    cmd_Str = f"ffmpeg -y -i {video_path} -ss {subtitle.start-timedelta(seconds=start_delta)} -to {subtitle.end-timedelta(seconds=end_delta)} -vn -ar 22050 -ac 1 -f {ext.replace('.','')} {output_fname}"
    subprocess.run(cmd_Str)
    return output_fname,subtitle.content
def extract_clip(video_path, evt:ass.section.EventsSection, output_dir=None,start_delta = 0.0,end_delta=0.0, ext:str=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    f = os.path.splitext(video_path)
    fname = f[0]
    if ext is None:
        ext = f[1]
    if output_dir is None:
        output_fname = fname+ext
    else:
        output_fname = output_dir + '/'+str(evt.start.seconds)+"-"+str(evt.end.seconds) + ext
    cmd_Str = f"ffmpeg -y -i {video_path} -ss {evt.start-timedelta(seconds=start_delta)} -to {evt.end-timedelta(seconds=end_delta)} -vn -f {ext.replace('.','')} {output_fname}"

    subprocess.run(cmd_Str)
    print(cmd_Str)
    return output_fname,evt.text
def extract_wav(video_path):
    '''
    extract wav audio from a video
    return result's filename
    '''
    fname = os.path.splitext(video_path)[0]
    ext = os.path.splitext(video_path)[1]
    cmd_str = f'ffmpeg -y -i {video_path} -vn -f wav {fname+".wav"}'
    subprocess.run(cmd_str)
    return fname+".wav"
def extract_vocal(track_path):
    '''
    extract vocal only audio from audio
    '''
    fname = os.path.splitext(track_path)[0]
    ext = os.path.splitext(track_path)[1]
    cmd_str = f'demucs --two-stem=vocals {track_path}'
    subprocess.run(cmd_str)
    fname_pure = fname[str(track_path).rfind('/')+1:]
    result_path = f'separated/htdemucs/{fname_pure}/vocals.wav'
    return result_path
def create_dataset(dataset_name,events,video_path,vocal = True):
    '''
    create dataset using ass subtitles
    '''
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)
    metadata = dataset_name+"/metadata.txt"
    f = open(metadata,'w')
    for evt in events:
        if 'jp' in str(evt.style):
            output_fname,txt = extract_clip(video_path,evt,dataset_name+"/wav",'.wav')
            if vocal:
                # output_fname = extract_wav(output_fname)
                vocal_output_fname = extract_vocal(output_fname)
                if os.path.exists(output_fname):
                    os.remove(output_fname)
                shutil.move(vocal_output_fname,output_fname)
            output_fname = os.path.relpath(output_fname,dataset_name)
            print(output_fname)
            f.write(output_fname+"|"+txt+"\n")
def create_dataset_srt(dataset_name,subtitles,video_path,vocal = True):
    '''
    create dataset using srt subtitles
    '''
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)
    metadata = dataset_name+"/metadata.txt"
    f = open(metadata,'w',encoding="utf-8")

    for subtitle in subtitles:
        # print('wtf')
        output_fname,txt = extract_clip_srt(video_path,subtitle,dataset_name+"/wav",'.wav')
        if vocal:
            # output_fname = extract_wav(output_fname)
            vocal_output_fname = extract_vocal(output_fname)
            if os.path.exists(output_fname):
                os.remove(output_fname)
            shutil.move(vocal_output_fname,output_fname)
        output_fname = os.path.relpath(output_fname,dataset_name)
        print(output_fname)
        print(txt)
        f.write(output_fname+"|"+txt+"\n")
def slice_by_silence(audio_path,out_path,min_silence_length=500,amplitude_diff_threshold=-35,padding=250):
    audio_path = pathlib.Path(audio_path)
    out_path = pathlib.Path(out_path)
    if not out_path.exists():
        os.makedirs(out_path)
    audio = AudioSegment.from_file(audio_path.resolve().__str__())
    # audio = AudioSegment.from_wav(audio_path.resolve().__str__())
    chunks = split_on_silence(audio,min_silence_length,amplitude_diff_threshold,padding)
    for i,chunk in enumerate(chunks):
        print(i)
        chunk:AudioSegment
        chunk.export(out_path.joinpath(f"{i}.wav").resolve().__str__(),format="wav")
# with open('misaka/sub.srt','r',encoding='utf-8') as f:
#     data = srt.parse(f.read())
# print('wtf')
# print(type(data))
# data = list(data)
# data = data[:2]
# for sub in data:
#     print(sub)
# create_dataset_srt('misaka_compass_ms',data,video_path="misaka/vid.mp3",vocal=False)
if __name__ =="__main__":
    # with open('kyouma.srt','r',encoding='utf-8') as f:
    #     data = srt.parse(f.read())
    # create_dataset_srt('kyouma_data',data,video_path="kyouma.mp3",vocal=False)
    audio_path = r"D:\pycharmWorkspace\vits-data\LycoRecoTest.wav"
    out_path = r"D:\pycharmWorkspace\vits-data\input\slices"
    # slice_by_silence(audio_path,out_path,800,-35,400)
    audio = AudioSegment.from_file(audio_path)
    playsound.playsound(audio_path)
    # play(audio)