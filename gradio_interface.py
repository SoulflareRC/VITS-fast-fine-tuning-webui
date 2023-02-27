import pathlib

import gradio as gr
import numpy as np
import inference_utils
from inference_utils import *
from training_utils import *
from finetune_speaker_kai import *
import torch

class gradio_ui(object):
    def __init__(self):
        # self.tts_model = TTSGenerator(model_path="trained_multiple_kai\G_latest.pth",config_path="trained_multiple_kai\config.json")
        self.whisper_model = WhisperTransciber()
        self.tts_model_list = {}
        self.tts_models_folder = "models"
        self.load_tts_models(self.tts_models_folder)
        self.current_model = self.tts_model_list[ list(self.tts_model_list.keys())[0] ]
        self.switch_tts_model(list(self.tts_model_list.keys())[0])

        #for training
        self.hps = utils.get_hparams_from_file(self.current_model['config_path'])
    def tts(self,text,speaker,language,speed):
        self.tts_model.speed = speed
        audio = self.tts_model.tts_infer(text,speaker,language)
        sampling_rate = audio.frame_rate
        arr = audio.get_array_of_samples()
        data = np.array(arr)
        return (sampling_rate,data)
    def test(self,input):
        print("Hello")
        print(input)
        return input
    def free_memory(self):
        self.tts_model = None
        self.whisper_model = None
        torch.cuda.empty_cache()
        gc.collect()
    def load_tts_models(self, path):
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
        return gr.Dropdown.update(choices=list(tts_models_list.keys()))
    def switch_tts_model(self,choice):
        if type(choice)==str and (choice in self.tts_model_list):
            model = self.tts_model_list[choice]
            self.tts_model = TTSGenerator(model_path=model['model_path'],config_path=model["config_path"])
            return gr.Dropdown.update(choices=self.tts_model.speakers)
        else:
            return gr.Dropdown.update()
    def train_tts_model(self,dataset_folder_path,eval_interval,epochs,learning_rate,batch_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8000'
        self.hps.train.eval_interval=int(eval_interval)
        self.hps.max_epochs = int(epochs)
        self.hps.train.learning_rate = learning_rate
        self.hps.train.batch_size = int(batch_size)
        self.hps.model_dir = os.path.join(dataset_folder_path,"trained_models")
        run(0,1,self.hps)
        print("Training finished!")


    def reload_models(self):
        self.tts_model = TTSGenerator(model_path="trained_multiple_kai\G_latest.pth",config_path="trained_multiple_kai\config.json")
        self.whisper_model = WhisperTransciber()
    def load_dataset(self,dataset_path):
        p = pathlib.Path(dataset_path)
        speakers = []
        if p.exists() and p.is_dir():
            fs = list(p.iterdir())
            annos_file = p.joinpath("annotations.txt")
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
        # test = ["masd","qwqe","qweqwe","addd","ppasp"]
        # eval_interval = gr.Number(label="Save every n steps", value=self.hps.train.eval_interval)
        # epochs = gr.Number(label="Epochs", value=self.hps.train.epochs)
        # learning_rate = gr.Number(label="Learning rate", value=self.hps.train.learning_rate)
        # batch_size = gr.Number(label="Batch size", value=self.hps.train.batch_size)
        return gr.List.update(value=speakers),\
               gr.Number.update (interactive=True, value=self.hps.train.eval_interval), \
                gr.Number.update(interactive=True, value=self.hps.train.epochs), \
            gr.Number.update(interactive=True,value=self.hps.train.learning_rate), \
            gr.Number.update(interactive=True, value=self.hps.train.batch_size)

    def interface(self):
        models_folder = gr.Textbox(label="Models folder")
        models_selection = gr.Dropdown(label="Models",choices=list(self.tts_model_list.keys()),value=list(self.tts_model_list.keys())[0])
        chara_selection = gr.Dropdown(label="Character",choices=self.tts_model.speakers,value=self.tts_model.speakers[-1])
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
        eval_interval = gr.Number(label="Save every n steps",value=self.hps.train.eval_interval,interactive=False)
        epochs = gr.Number(label="Epochs",value=self.hps.train.epochs,interactive=False)
        learning_rate = gr.Number(label="Learning rate",value=self.hps.train.learning_rate,interactive=False)
        batch_size = gr.Number(label="Batch size",value=self.hps.train.batch_size,interactive=False)
        train_btn = gr.Button(value="Start training!",variant="primary",interactive=False)
        train_hint = gr.Markdown(value="It is recommended to free memory before training.")


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
                with gr.Tab("Preprocess"):
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

            hibernate_btn.render()
            reload_btn.render()

            models_folder.submit(fn=self.load_tts_models, inputs=models_folder, outputs=models_selection)
            models_selection.change(fn=self.switch_tts_model,inputs=models_selection,outputs=chara_selection)

            generate_audio.click(fn=self.tts,inputs=[speech_prompt,chara_selection,language_selection,speed_slider],outputs=output_audio)

            dataset_load_btn.click(fn=self.load_dataset,inputs=dataset_folder_path,outputs=[speakers_list,eval_interval,epochs,learning_rate,batch_size])
            speakers_list.set_event_trigger(event_name="click",fn=self.test,inputs=speakers_list,outputs=speakers_list)
            train_btn.click(fn=self.train_tts_model,inputs=[dataset_folder_path, eval_interval,epochs,learning_rate,batch_size])


            hibernate_btn.click(fn=self.free_memory)
            reload_btn.click(fn=self.reload_models)
        demo.launch(server_port=2000)
gradio_ui().interface()