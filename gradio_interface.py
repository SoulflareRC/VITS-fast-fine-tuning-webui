import pathlib
import tempfile

import gradio as gr
import numpy as np
import inference_utils
from inference_utils import *
from training_utils import *
from finetune_speaker_kai import *
import torch
import time
import os
from scipy.io.wavfile import write
import shutil
from voiceline_extraction import *

class control(object):
    def __init__(self):
        # self.tts_model = TTSGenerator(model_path="trained_multiple_kai\G_latest.pth",config_path="trained_multiple_kai\config.json")
        self.whisper_model =None # WhisperTransciber()
        self.tts_model_list = {}
        self.tts_models_folder = "models"
        self.tts_model = None
        # don't load model on launch, only load when needed
        # self.load_tts_models(self.tts_models_folder)
        # self.current_model = self.tts_model_list[ list(self.tts_model_list.keys())[0] ]
        # self.switch_tts_model(list(self.tts_model_list.keys())[0])

        #for training
        # self.hps = utils.get_hparams_from_file(self.current_model['config_path'])
    def write_audio(self,data, fname):
        rate = 44100
        rate /= 2
        rate = int(rate)
        scaled = np.int16(data / np.max(np.abs(data)) * 32767)
        # scaled = np.int16(data)
        write(fname, rate, scaled)
    def tts(self,text,speaker,language,speed)->AudioSegment:
        '''
        takes in text, speaker,language, speed
        return audio(AudioSegment)
        '''
        self.tts_model.speed = speed
        audio:AudioSegment = self.tts_model.tts_infer(text,speaker,language)
        # print("Done!")
        # sampling_rate = audio.frame_rate
        # arr = audio.get_array_of_samples()
        # data = np.array(arr)
        # return (sampling_rate,data)
        audio_folder = pathlib.Path("output")
        if not audio_folder.exists():
            os.makedirs(audio_folder)
        timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
        fname =audio_folder.joinpath(f"{timestamp}.wav")
        self.write_audio(audio.get_array_of_samples(),fname)
        return audio,fname
    def test(self,input):
        print("Hello")
        print(input)
        return input
    def free_memory(self):
        '''
        unload whisper and tts models, free memory
        '''
        self.tts_model = None
        self.whisper_model = None
        torch.cuda.empty_cache()
        gc.collect()
    def load_tts_models(self, path):
        '''
        takes in models folder path,returns a list of models including model name,model and config's path
        (doesn't automatically load any model)
        load model by calling switch_tts_model
        '''
        models_folder = pathlib.Path(path)
        tts_models_list = {}
        if models_folder.exists() and models_folder.is_dir():
            fs = list(models_folder.iterdir())
            for f in fs:
                if f.suffix ==".pth":
                    config_file = f.parent.joinpath(f.stem+".json")
                    if config_file.exists():
                        #both config and checkpoint exists, load this
                        name = f.stem
                        tts_models_list[name] = {
                            "name":name,
                            "model_path":f.resolve(),
                            "config_path":config_file.resolve()
                        }
        self.tts_model_list = tts_models_list
        return self.tts_model_list
        # return gr.Dropdown.update(choices=list(tts_models_list.keys()))
    def switch_tts_model(self,choice):
        '''
        takes in model name, load the TTS model according to config and model path
         return speakers list
        '''
        if type(choice)==str and (choice in self.tts_model_list):
            model = self.tts_model_list[choice]
            self.tts_model = TTSGenerator(model_path=model['model_path'],config_path=model["config_path"])
            return self.tts_model.speakers
        else:
            return []
    def train_tts_model(self,dataset_folder_path,eval_interval,epochs,learning_rate,batch_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '7000'
        self.hps.train.eval_interval=int(eval_interval)
        self.hps.max_epochs = int(epochs)
        self.hps.train.learning_rate = learning_rate
        self.hps.train.batch_size = int(batch_size)
        self.hps.model_dir = os.path.join(dataset_folder_path,"trained_models")
        run(0,1,self.hps)
        print("Training finished!")
        return self.hps.model_dir
    def reload_models(self):
        self.tts_model = TTSGenerator(model_path="trained_multiple_kai\G_latest.pth",config_path="trained_multiple_kai\config.json")
        self.whisper_model = WhisperTransciber()
    def load_dataset(self,dataset_path):
        p = pathlib.Path(dataset_path)
        speakers = []
        if p.exists() and p.is_dir():
            fs = list(p.iterdir())
            annos_file = p.joinpath("annotations.txt")
            if self.whisper_model is None:
                print("Loading Whisper model")
                self.whisper_model = WhisperTransciber()
            if not annos_file.exists():
                print(f"Annotation file doesn't exist,using whisper transcribe to generate annotations...")

                self.whisper_model.transcribe_dataset(dataset_path)
            self.whisper_model.annotations_train_val_split(dataset_path)
            for f in fs:
                if f.is_dir():
                    cur_speaker = [f.stem]
                    cur_speaker.append(str(len(list(f.iterdir())))+" voice lines")
                    speakers.append(cur_speaker)
            training_config_file = p.joinpath("modified_finetune_speaker.json")
            self.hps = utils.get_hparams_from_file(training_config_file.resolve().__str__())
        print(f"Found speakers {speakers}")
        return speakers
class gradio_ui(object):
    def __init__(self):
        self.control_model = control()
        self.control_model.load_tts_models("models")
        self.control_model.switch_tts_model(list(self.control_model.tts_model_list.keys())[0])
        self.max_voiceline_num = 10000
        # view dataset
        self.charas = {
            "None": []
        }
    def load_dataset2(self,path):
        charas = {
            "All": [],
            "None": []
        }
        folder = pathlib.Path(path)
        if folder.exists() and folder.is_dir():
            fs = [f for f in list(folder.iterdir())]
            for f in fs:
                print(f)
                if f.is_file() and f.suffix == ".wav":
                    print(f)
                    aud = AudioSegment.from_file(f.resolve().__str__())
                    audio = {
                        "name": f.name,
                        "path": f,
                        "audio": aud
                    }
                    charas['None'].append(audio)
                    charas['All'].append(audio)
                if f.is_dir():
                    chara_name = f.stem
                    chara_fs = [f for f in list(f.iterdir()) if f.suffix == ".wav"]
                    charas[chara_name] = []
                    for chara_f in chara_fs:
                        chara_aud = AudioSegment.from_file(chara_f.resolve().__str__())
                        chara_audio = {
                            "name": chara_f.relative_to(folder),
                            "path": chara_f,
                            "audio": chara_aud
                        }
                        charas[chara_name].append(chara_audio)
                        charas['All'].append(chara_audio)
        return charas
    def load_voicelines(self,path):

        self.charas = self.load_dataset2(path)
        print(self.charas)
        chara_names = list(self.charas.keys())
        current_chara = chara_names[0]
        all_voicelines = self.charas['All']
        print(f"Found {len(all_voicelines)} voicelines")
        ret_voicelines = []
        for a in all_voicelines:
            aud: AudioSegment = a['audio']
            name = a['name']
            path = a['path']
            ret_voicelines.append(gr.Audio.update(value=path, visible=True, label=name))
        while len(ret_voicelines) < self.max_voiceline_num:
            ret_voicelines.append(gr.Audio.update(visible=False, value=None))
        ret_voicelines.append(gr.Dropdown.update(choices=chara_names, value=current_chara, interactive=True))
        return ret_voicelines
        # return gr.CheckboxGroup.update(choices= [a['name'] for a in all_voicelines], value=[a['name'] for a in charas[current_chara]]),\
        #         gr.Dropdown.update(choices=chara_names,value=current_chara,interactive=True)
    def switch_chara(self,chara):
        print("Switched to ", chara)
        charas = self.charas
        ret_voicelines = []
        for key in charas.keys():
            print(f"{len(charas[key])} voicelines for {key}")
        chara_voicelines = charas[chara]
        for a in chara_voicelines:
            aud: AudioSegment = a['audio']
            name = a['name']
            path = a['path']
            ret_voicelines.append(gr.Audio.update(value=path, visible=True, label=name))
        while len(ret_voicelines) < self.max_voiceline_num:
            ret_voicelines.append(gr.Audio.update(visible=False, value=None))
        return ret_voicelines

    def tts(self, text, speaker, language, speed):
        audio,path = self.control_model.tts(text,speaker,language,speed)
        audio:AudioSegment
        return (audio.frame_rate,np.array(audio.get_array_of_samples()))
    def free_memory(self):
        self.control_model.free_memory()
        print("Memory freed!")

    def load_tts_models(self, path):
        self.control_model.load_tts_models(path)
        return gr.Dropdown.update(choices=list(self.control_model.tts_models_list.keys()))
    def switch_tts_model(self, choice):
        speakers = self.control_model.switch_tts_model(choice)
        return gr.Dropdown.update(choices=speakers)

    def train_tts_model(self, dataset_folder_path, eval_interval, epochs, learning_rate, batch_size):
        self.control_model.train_tts_model(dataset_folder_path,eval_interval,epochs,learning_rate,batch_size)
        print("Training finished!")

    def reload_models(self):
        self.control_model.reload_models()

    def load_dataset(self, dataset_path):
        speakers = self.control_model.load_dataset(dataset_path)
        return gr.List.update(value=speakers),\
                       gr.Number.update (interactive=True, value=self.control_model.hps.train.eval_interval), \
                        gr.Number.update(interactive=True, value=self.control_model.hps.train.epochs), \
                    gr.Number.update(interactive=True,value=self.control_model.hps.train.learning_rate), \
                    gr.Number.update(interactive=True, value=self.control_model.hps.train.batch_size)
    def extract_voiceline_from_sub(self,vid_path,sub_path,out_path,format,start_delta,end_delta):
        print("Extract voicelines from subtitles file.")
        vid = pathlib.Path(vid_path)
        sub = pathlib.Path(sub_path)
        out = pathlib.Path(out_path)
        if not out.exists():
            os.makedirs(out)
        if vid.exists() and sub.exists():
            audio_path = extract_wav(vid.resolve().__str__())
            if format == "ass":
                with open(sub,encoding="utf_8_sig") as f:
                    sub = ass.parse(f)
                evts = sub.events
                for evt in evts:
                    extract_clip(video_path=vid_path,evt=evt,output_dir=out_path,start_delta=start_delta,end_delta=end_delta,ext=".wav")
            elif format == "srt":
                with open(sub, encoding="utf_8_sig") as f:
                    evts = srt.parse(f.read())
                for sub in evts:
                    extract_clip_srt(video_path=vid_path,subtitle=sub,output_dir=out_path,ext=".wav",start_delta=start_delta,end_delta=end_delta)
            return f"Successfully extracted {len(list(out.iterdir()))} voicelines to {out.resolve()}"
        return "Failed to extract voicelines"
    def upload_to_text(self,file:tempfile.TemporaryFile):
        print(file.name)

        return file.name
    def slice_by_silence(self,audio_path,out_path,length,threshold,keep):
        audio_path = pathlib.Path(audio_path)
        if not audio_path.exists():
            return "Failed to extract voicelines"
        out = pathlib.Path(out_path)
        slice_by_silence(audio_path,out_path,int(length),int(threshold),int(keep))
        return f"Successfully extracted {len(list(out.iterdir()))} voicelines to {out.resolve()}"

    def edit_load_dataset(self,path):
        charas = {
            "All": [],
            "None": []
        }
        folder = pathlib.Path(path)
        if folder.exists() and folder.is_dir():
            fs = [f for f in list(folder.iterdir())]
            for f in fs:
                print(f)
                if f.is_file() and f.suffix == ".wav":
                    print(f)
                    aud = AudioSegment.from_file(f.resolve().__str__())
                    audio = {
                        "name": f.name,
                        "path": f,
                        "audio": aud
                    }
                    charas['None'].append(audio)
                    charas['All'].append(audio)
                if f.is_dir():
                    chara_name = f.stem
                    chara_fs = [f for f in list(f.iterdir()) if f.suffix == ".wav"]
                    charas[chara_name] = []
                    for chara_f in chara_fs:
                        chara_aud = AudioSegment.from_file(chara_f.resolve().__str__())
                        chara_audio = {
                            "name": chara_f.relative_to(folder),
                            "path": chara_f,
                            "audio": chara_aud
                        }
                        charas[chara_name].append(chara_audio)
                        charas['All'].append(chara_audio)
        return charas

    def edit_load_dataset_gr(self,path):
        '''
        output to
        hint(str),dropdown1,dropdown2
        checkboxgroup1,checkboxgroup2
        '''
        global charas
        charas = self.edit_load_dataset(path)
        all_voicelines = charas['All']

        hint_message = "Successfully loaded dataset!"
        return gr.Textbox.update(value=hint_message), \
            gr.Dropdown.update(choices=list(charas.keys()), value="All", interactive=True), \
            gr.Dropdown.update(choices=list(charas.keys()), value="All", interactive=True), \
            gr.CheckboxGroup.update(choices=[a['name'] for a in all_voicelines],
                                    label=f"{len(all_voicelines)} voicelines", interactive=True), \
            gr.CheckboxGroup.update(choices=[a['name'] for a in all_voicelines],
                                    label=f"{len(all_voicelines)} voicelines", interactive=True)
        # return gr.CheckboxGroup.update(choices= [a['name'] for a in all_voicelines], value=[a['name'] for a in charas[current_chara]]),\
        #         gr.Dropdown.update(choices=chara_names,value=current_chara,interactive=True)

    def move_file(self,src_file, dest_folder):
        src_path = pathlib.Path(src_file)
        dest_path = pathlib.Path(dest_folder) / src_path.name

        if dest_path.exists():
            dest_path = dest_path.with_name(f"{src_path.stem}_1{src_path.suffix}")

        shutil.move(str(src_path), str(dest_path))
        return str(dest_path)

    def edit_switch_chara(self,chara):
        print("Switched to ", chara)
        if type(chara) == str and chara in charas.keys():
            for key in charas.keys():
                print(f"{len(charas[key])} voicelines for {key}")
            chara_voicelines = charas[chara]
            return gr.CheckboxGroup.update(choices=[a['name'] for a in chara_voicelines],label=f"{len(chara_voicelines)} voicelines")
        else:
            return gr.CheckboxGroup.update()

    def edit_move_audios(self,dataset_path, src_selection: list[str], src_chara, dest_chara):
        '''
        return 2 checkboxgroups for src and dest
        '''
        if src_chara == dest_chara or dest_chara == "All":
            # if src is dest or dest is all then do nothing
            return gr.CheckboxGroup.update(), gr.CheckboxGroup.update()
        else:
            # move selected audio files from source folder to dest folder
            dataset_folder = pathlib.Path(dataset_path)
            target_folder = dataset_folder if dest_chara == "None" else dataset_folder.joinpath(dest_chara)
            for f in src_selection:
                source = dataset_folder.joinpath(f)
                self.move_file(source, target_folder)
            global charas
            charas = self.edit_load_dataset(dataset_path)
            src_chara_voicelines = charas[src_chara]
            dest_chara_voicelines = charas[dest_chara]
            return gr.CheckboxGroup.update(choices=[a['name'] for a in src_chara_voicelines],
                                           label=f"{len(src_chara_voicelines)} voicelines"), \
                gr.CheckboxGroup.update(choices=[a['name'] for a in dest_chara_voicelines],
                                        label=f"{len(dest_chara_voicelines)} voicelines")
    def interface(self):
        models_folder = gr.Textbox(label="Models folder")
        models_selection = gr.Dropdown(label="Models",choices=list(self.control_model.tts_model_list.keys()),value=list(self.control_model.tts_model_list.keys())[0])
        chara_selection = gr.Dropdown(label="Character",choices=self.control_model.tts_model.speakers,value=self.control_model.tts_model.speakers[-1])
        language_selection = gr.Dropdown(label="Language",choices=inference_utils.lang,value=inference_utils.lang[0])
        speed_slider = gr.Slider(minimum=0.1, maximum=5, value=1.0)
        output_audios = []
        max_audio_output = 10
        for i in range(max_audio_output):
            output_audios.append(gr.Audio(visible=False, show_label=False))
        speech_prompt = gr.TextArea(label="Text",value="こんにちは")
        generate_audio = gr.Button(value="Generate",variant="primary")
        output_audio = gr.Audio(label="Speech")

        # training tab
        '''
        Doesn't need an existing config file, this will just take the dataset and then
        1. whisper preprocess
        2. generate annotations
        after pressing training button:
        1. generate config according to settings
        2. start training
        '''
        dataset_folder_path = gr.Textbox(label="Dataset",placeholder="Path to dataset's folder")
        dataset_load_btn = gr.Button(value="Load Dataset",variant="primary")
        speakers_list = gr.List(label="Speakers")
        training_config_path = gr.Textbox(label="Config Path",placeholder="Path to training config")
        model_output_dir = gr.Textbox(label="Output Folder",placeholder="Trained models will be saved here")
        hibernate_btn = gr.Button(value="Hibernate (free memory)")
        reload_btn = gr.Button(value="Reload models (Use this after freeing memory)")

        # training config
        eval_interval = gr.Number(label="Save every n steps",value=-1,interactive=False)
        epochs = gr.Number(label="Epochs",value=-1,interactive=False)
        learning_rate = gr.Number(label="Learning rate",value=-1,interactive=False)
        batch_size = gr.Number(label="Batch size",value=-1,interactive=False)
        train_btn = gr.Button(value="Start training!",variant="primary",interactive=False)
        train_hint = gr.Markdown(value="It is recommended to free memory before training.")


        dataset_folder_path2 = gr.Textbox(label="Dataset Path", placeholder="Path to dataset")
        dataset_load_btn2 = gr.Button(value="Load dataset")

        dataset_chara_selection = gr.Dropdown(label="Characters", choices=list(self.charas.keys()), value="None",
                                              interactive=True)
        dataset_chara_name = gr.Textbox(label="Add a new character")
        dataset_chara_submit = gr.Button(value="Add Character")

        dataset_voicelines = gr.CheckboxGroup(label="All voicelines", interactive=True)

        dataset_voicelines = []
        for i in range(self.max_voiceline_num):
            dataset_voicelines.append(gr.Audio(visible=False))


        #Preprocessing tab
        preprocess_sub_vid_path = gr.Textbox(label="Path to video")
        # preprocess_sub_vid_path_upload = gr.UploadButton()
        preprocess_sub_sub_path = gr.Textbox(label="Path to subtitle",placeholder="Path to the subtitles file, support ass and srt format")
        # preprocess_sub_sub_path_upload = gr.UploadButton()
        preprocess_sub_out_path=gr.Textbox(label="Path to output folder")
        # preprocess_sub_out_path_upload = gr.UploadButton(file_count="directory")
        preprocess_sub_sub_format = gr.Radio(choices=['ass','srt'],value='ass',label="subtitles file format")
        preprocess_sub_start_delta = gr.Number(label="Shift start timestamp",value=0.5)
        preprocess_sub_end_delta = gr.Number(label="Shift end timestamp", value=0.5)
        preprocess_sub_submit = gr.Button(value="Extract voicelines")
        preprocess_sub_progress = gr.Progress(track_tqdm=True)
        preprocess_sub_hint = gr.Textbox(label="Status",interactive=False)

        preprocess_sil_aud_path = gr.Textbox(label="Path to audio file")
        preprocess_sil_out_path = gr.Textbox(label="Path to output folder")
        preprocess_sil_length = gr.Number(label="Minimum silence length(ms)",value=500)
        preprocess_sil_threshold = gr.Number(label="Volume change threshold",value=-35)
        preprocess_sil_keep = gr.Number(label="Silence to keep(ms)",value=250)
        preprocess_sil_submit = gr.Button(value="Extract!",variant="primary")
        preprocess_sil_hint = gr.Textbox(label="Status",interactive=False)

        # edit dataset
        edit_dataset_path = gr.Textbox(label="Dataset Folder Path")
        edit_dataset_load_btn = gr.Button(value="Load dataset")
        edit_dataset_create_btn = gr.Button(value="Create dataset")
        edit_dataset_hint = gr.Textbox(interactive=False, label="Message")

        edit_move_select_btn_l = gr.Button(value="Move selected to right")
        edit_chara_selection_l = gr.Dropdown(label="Characters")
        edit_audio_display_l = gr.CheckboxGroup(label="Voice lines")

        edit_move_select_btn_r = gr.Button(value="Move selected to left")
        edit_chara_selection_r = gr.Dropdown(label="Characters")
        edit_audio_display_r = gr.CheckboxGroup(label="Voice lines")

        with gr.Blocks() as demo:
            with gr.Tab("Inference"):
                with gr.Column(scale=1):
                    # models_folder.render()
                    models_selection.render()
                    with gr.Row():
                        chara_selection.render()
                        language_selection.render()
                    speed_slider.render()
                with gr.Column(scale=1):
                    speech_prompt.render()
                    generate_audio.render()
                    for audio in output_audios:
                        audio.render()
                    output_audio.render()
            with gr.Tab("Train"):
                with gr.Tab("View Dataset"):
                    with gr.Column():
                        with gr.Row():
                            dataset_folder_path2.render()
                            dataset_load_btn2.render()
                        with gr.Box():
                            dataset_chara_name.render()
                            dataset_chara_submit.render()
                            dataset_chara_selection.render()
                            # test_dataset.render()
                        with gr.Row():
                            # dataset_voicelines.render()
                            for a in dataset_voicelines:
                                a.render()
                    dataset_load_btn2.click(fn=self.load_voicelines, inputs=dataset_folder_path2,
                                           outputs=dataset_voicelines + [dataset_chara_selection])
                    dataset_chara_selection.change(fn=self.switch_chara, inputs=dataset_chara_selection,
                                                   outputs=dataset_voicelines)
                    pass
                with gr.Tab("Train"):
                    with gr.Column():
                        with gr.Row():
                            dataset_folder_path.render()
                            dataset_load_btn.render()
                    speakers_list.render()
                    training_config_path.render()
                    model_output_dir.render()
                    with gr.Accordion(label="Training Configs"):
                        eval_interval.render()
                        epochs.render()
                        learning_rate.render()
                        batch_size.render()
                    train_btn.render()
                    train_hint.render()
                with gr.Tab("Preprocessing"):
                    with gr.Tab("Extract from subtitles file"):
                        with gr.Row():
                            preprocess_sub_vid_path.render()
                            # preprocess_sub_vid_path_upload.render()
                        with gr.Row():
                            preprocess_sub_sub_path.render()
                            # preprocess_sub_sub_path_upload.render()
                        with gr.Row():
                            preprocess_sub_out_path.render()
                            # preprocess_sub_out_path_upload.render()
                        preprocess_sub_sub_format.render()
                        with gr.Row():
                            preprocess_sub_start_delta.render()
                            preprocess_sub_end_delta.render()
                        preprocess_sub_submit.render()
                        preprocess_sub_hint.render()
                        # preprocess_sub_progress.ren
                    with gr.Tab("Slice by silence"):
                        preprocess_sil_aud_path.render()
                        preprocess_sil_out_path.render()
                        with gr.Box():
                            preprocess_sil_length.render()
                            preprocess_sil_threshold.render()
                            preprocess_sil_keep.render()
                        preprocess_sil_submit.render()
                        preprocess_sil_hint.render()
            with gr.Tab("Edit dataset"):
                with gr.Row():
                    edit_dataset_path.render()
                    with gr.Column():
                        edit_dataset_load_btn.render()
                        edit_dataset_create_btn.render()
                with gr.Row():
                    edit_dataset_hint.render()
                with gr.Row():
                    with gr.Column():
                        edit_move_select_btn_l.render()
                        edit_chara_selection_l.render()
                        edit_audio_display_l.render()
                    with gr.Column():
                        edit_move_select_btn_r.render()
                        edit_chara_selection_r.render()
                        edit_audio_display_r.render()

                edit_dataset_load_btn.click(fn=self.edit_load_dataset_gr, inputs=edit_dataset_path,
                                            outputs=[edit_dataset_hint, edit_chara_selection_l, edit_chara_selection_r,
                                                     edit_audio_display_l, edit_audio_display_r])

                edit_chara_selection_l.change(fn=self.edit_switch_chara, inputs=edit_chara_selection_l,
                                              outputs=edit_audio_display_l)
                edit_chara_selection_r.change(fn=self.edit_switch_chara, inputs=edit_chara_selection_r,
                                              outputs=edit_audio_display_r)

                edit_move_select_btn_l.click(fn=self.edit_move_audios,
                                             inputs=[edit_dataset_path, edit_audio_display_l, edit_chara_selection_l,
                                                     edit_chara_selection_r],
                                             outputs=[edit_audio_display_l, edit_audio_display_r])
                edit_move_select_btn_r.click(fn=self.edit_move_audios,
                                             inputs=[edit_dataset_path, edit_audio_display_r, edit_chara_selection_r,
                                                     edit_chara_selection_l],
                                             outputs=[edit_audio_display_r, edit_audio_display_l])
            hibernate_btn.render()
            reload_btn.render()

            models_folder.submit(fn=self.load_tts_models, inputs=models_folder, outputs=models_selection)
            models_selection.change(fn=self.switch_tts_model,inputs=models_selection,outputs=chara_selection)

            generate_audio.click(fn=self.tts,inputs=[speech_prompt,chara_selection,language_selection,speed_slider],outputs=output_audio)

            dataset_load_btn.click(fn=self.load_dataset,inputs=dataset_folder_path,outputs=[speakers_list,eval_interval,epochs,learning_rate,batch_size])
            # speakers_list.set_event_trigger(event_name="click",fn=self.test,inputs=speakers_list,outputs=speakers_list)
            train_btn.click(fn=self.train_tts_model,inputs=[dataset_folder_path, eval_interval,epochs,learning_rate,batch_size])

            # preprocess_sub_vid_path_upload.upload(fn=self.upload_to_text, inputs=preprocess_sub_vid_path_upload,outputs=preprocess_sub_vid_path)
            # preprocess_sub_sub_path_upload.upload(fn=self.upload_to_text, inputs=preprocess_sub_sub_path_upload,outputs=preprocess_sub_sub_path)
            # preprocess_sub_out_path_upload.upload(fn=self.upload_to_text, inputs=preprocess_sub_out_path_upload,outputs=preprocess_sub_out_path)
            preprocess_sub_submit.click(fn=self.extract_voiceline_from_sub,
                                        inputs=[preprocess_sub_vid_path,
                                                preprocess_sub_sub_path,
                                                preprocess_sub_out_path,
                                                preprocess_sub_sub_format,
                                                preprocess_sub_start_delta,
                                                preprocess_sub_end_delta],outputs=preprocess_sub_hint)
            preprocess_sil_submit.click(fn=self.slice_by_silence,inputs=[preprocess_sil_aud_path,preprocess_sil_out_path, preprocess_sil_length,preprocess_sil_threshold,preprocess_sil_keep],outputs=preprocess_sil_hint)
            hibernate_btn.click(fn=self.free_memory)
            reload_btn.click(fn=self.reload_models)
        demo.launch(server_port=2000)
# class gradio_ui(object):
#     def __init__(self):
#         # self.tts_model = TTSGenerator(model_path="trained_multiple_kai\G_latest.pth",config_path="trained_multiple_kai\config.json")
#         self.whisper_model = WhisperTransciber()
#         self.tts_model_list = {}
#         self.tts_models_folder = "models"
#         self.load_tts_models(self.tts_models_folder)
#         self.current_model = self.tts_model_list[ list(self.tts_model_list.keys())[0] ]
#         self.switch_tts_model(list(self.tts_model_list.keys())[0])
#
#         #for training
#         self.hps = utils.get_hparams_from_file(self.current_model['config_path'])
#     def tts(self,text,speaker,language,speed):
#         self.tts_model.speed = speed
#         audio = self.tts_model.tts_infer(text,speaker,language)
#         sampling_rate = audio.frame_rate
#         arr = audio.get_array_of_samples()
#         data = np.array(arr)
#         return (sampling_rate,data)
#     def test(self,input):
#         print("Hello")
#         print(input)
#         return input
#     def free_memory(self):
#         self.tts_model = None
#         self.whisper_model = None
#         torch.cuda.empty_cache()
#         gc.collect()
#     def load_tts_models(self, path):
#         models_folder = pathlib.Path(path)
#         tts_models_list = {}
#         if models_folder.exists() and models_folder.is_dir():
#             fs = list(models_folder.iterdir())
#             for f in fs:
#                 if f.suffix ==".pth":
#                     config_file = f.parent.joinpath(f.stem+".json")
#                     if config_file.exists():
#                         #both config and checkpoint exists, load this
#                         name = f.stem
#                         tts_models_list[name] = {
#                             "name":name,
#                             "model_path":f.resolve(),
#                             "config_path":config_file.resolve()
#                         }
#         self.tts_model_list = tts_models_list
#         return gr.Dropdown.update(choices=list(tts_models_list.keys()))
#     def switch_tts_model(self,choice):
#         if type(choice)==str and (choice in self.tts_model_list):
#             model = self.tts_model_list[choice]
#             self.tts_model = TTSGenerator(model_path=model['model_path'],config_path=model["config_path"])
#             return gr.Dropdown.update(choices=self.tts_model.speakers)
#         else:
#             return gr.Dropdown.update()
#     def train_tts_model(self,dataset_folder_path,eval_interval,epochs,learning_rate,batch_size):
#         os.environ['MASTER_ADDR'] = 'localhost'
#         os.environ['MASTER_PORT'] = '8000'
#         self.hps.train.eval_interval=int(eval_interval)
#         self.hps.max_epochs = int(epochs)
#         self.hps.train.learning_rate = learning_rate
#         self.hps.train.batch_size = int(batch_size)
#         self.hps.model_dir = os.path.join(dataset_folder_path,"trained_models")
#         run(0,1,self.hps)
#         print("Training finished!")
#
#
#     def reload_models(self):
#         self.tts_model = TTSGenerator(model_path="trained_multiple_kai\G_latest.pth",config_path="trained_multiple_kai\config.json")
#         self.whisper_model = WhisperTransciber()
#     def load_dataset(self,dataset_path):
#         p = pathlib.Path(dataset_path)
#         speakers = []
#         if p.exists() and p.is_dir():
#             fs = list(p.iterdir())
#             annos_file = p.joinpath("annotations.txt")
#             if not annos_file.exists():
#                 print(f"Annotation file doesn't exist,using whisper transcribe to generate annotations...")
#                 self.whisper_model.transcribe_dataset(dataset_path)
#             self.whisper_model.annotations_train_val_split(dataset_path)
#             for f in fs:
#                 if f.is_dir():
#                     cur_speaker = [f.stem]
#                     cur_speaker.append(str(len(list(f.iterdir())))+" voice lines")
#                     speakers.append(cur_speaker)
#             training_config_file = p.joinpath("modified_finetune_speaker.json")
#             self.hps = utils.get_hparams_from_file(training_config_file.resolve().__str__())
#         print(f"Found speakers {speakers}")
#         # test = ["masd","qwqe","qweqwe","addd","ppasp"]
#         # eval_interval = gr.Number(label="Save every n steps", value=self.hps.train.eval_interval)
#         # epochs = gr.Number(label="Epochs", value=self.hps.train.epochs)
#         # learning_rate = gr.Number(label="Learning rate", value=self.hps.train.learning_rate)
#         # batch_size = gr.Number(label="Batch size", value=self.hps.train.batch_size)
#         return gr.List.update(value=speakers),\
#                gr.Number.update (interactive=True, value=self.hps.train.eval_interval), \
#                 gr.Number.update(interactive=True, value=self.hps.train.epochs), \
#             gr.Number.update(interactive=True,value=self.hps.train.learning_rate), \
#             gr.Number.update(interactive=True, value=self.hps.train.batch_size)
#
#     def interface(self):
#         models_folder = gr.Textbox(label="Models folder")
#         models_selection = gr.Dropdown(label="Models",choices=list(self.tts_model_list.keys()),value=list(self.tts_model_list.keys())[0])
#         chara_selection = gr.Dropdown(label="Character",choices=self.tts_model.speakers,value=self.tts_model.speakers[-1])
#         language_selection = gr.Dropdown(label="Language",choices=inference_utils.lang,value=inference_utils.lang[0])
#         speed_slider = gr.Slider(minimum=0.1, maximum=5, value=1.0)
#         output_audios = []
#         max_audio_output = 10
#         for i in range(max_audio_output):
#             output_audios.append(gr.Audio(visible=False, show_label=False))
#         speech_prompt = gr.TextArea(label="Text",value="こんにちは")
#         generate_audio = gr.Button(value="Generate",variant="primary")
#         output_audio = gr.Audio(label="Speech")
#
#         # training tab
#         '''
#         Doesn't need an existing config file, this will just take the dataset and then
#         1. whisper preprocess
#         2. generate annotations
#         after pressing training button:
#         1. generate config according to settings
#         2. start training
#         '''
#         dataset_folder_path = gr.Textbox(label="Dataset",placeholder="Path to dataset's folder")
#         dataset_load_btn = gr.Button(value="Load Dataset",variant="primary")
#         speakers_list = gr.List(label="Speakers")
#         training_config_path = gr.Textbox(label="Config Path",placeholder="Path to training config")
#         model_output_dir = gr.Textbox(label="Output Folder",placeholder="Trained models will be saved here")
#         hibernate_btn = gr.Button(value="Hibernate (free memory)")
#         reload_btn = gr.Button(value="Reload models (Use this after freeing memory)")
#
#         # training config
#         eval_interval = gr.Number(label="Save every n steps",value=self.hps.train.eval_interval,interactive=False)
#         epochs = gr.Number(label="Epochs",value=self.hps.train.epochs,interactive=False)
#         learning_rate = gr.Number(label="Learning rate",value=self.hps.train.learning_rate,interactive=False)
#         batch_size = gr.Number(label="Batch size",value=self.hps.train.batch_size,interactive=False)
#         train_btn = gr.Button(value="Start training!",variant="primary",interactive=False)
#         train_hint = gr.Markdown(value="It is recommended to free memory before training.")
#
#
#         with gr.Blocks() as demo:
#             with gr.Tab("Inference"):
#                 with gr.Column(scale=1):
#                     # models_folder.render()
#                     models_selection.render()
#                     with gr.Row():
#                         chara_selection.render()
#                         language_selection.render()
#                     speed_slider.render()
#                 with gr.Column(scale=1):
#                     speech_prompt.render()
#                     generate_audio.render()
#                     for audio in output_audios:
#                         audio.render()
#                     output_audio.render()
#             with gr.Tab("Train"):
#                 with gr.Tab("Preprocess"):
#                     # with gr.Box():
#                     #     gr.Dataset([gr.Audio(),gr.Audio,gr.Audio])
#                     pass
#                 with gr.Tab("Train"):
#                     with gr.Column():
#                         with gr.Row():
#                             dataset_folder_path.render()
#                             dataset_load_btn.render()
#                     speakers_list.render()
#                     training_config_path.render()
#                     model_output_dir.render()
#                     with gr.Accordion(label="Training Configs"):
#                         eval_interval.render()
#                         epochs.render()
#                         learning_rate.render()
#                         batch_size.render()
#                     train_btn.render()
#                     train_hint.render()
#
#             hibernate_btn.render()
#             reload_btn.render()
#
#             models_folder.submit(fn=self.load_tts_models, inputs=models_folder, outputs=models_selection)
#             models_selection.change(fn=self.switch_tts_model,inputs=models_selection,outputs=chara_selection)
#
#             generate_audio.click(fn=self.tts,inputs=[speech_prompt,chara_selection,language_selection,speed_slider],outputs=output_audio)
#
#             dataset_load_btn.click(fn=self.load_dataset,inputs=dataset_folder_path,outputs=[speakers_list,eval_interval,epochs,learning_rate,batch_size])
#             speakers_list.set_event_trigger(event_name="click",fn=self.test,inputs=speakers_list,outputs=speakers_list)
#             train_btn.click(fn=self.train_tts_model,inputs=[dataset_folder_path, eval_interval,epochs,learning_rate,batch_size])
#
#
#             hibernate_btn.click(fn=self.free_memory)
#             reload_btn.click(fn=self.reload_models)
#         demo.launch(server_port=2000)
if __name__ == "__main__":
    gradio_ui().interface()