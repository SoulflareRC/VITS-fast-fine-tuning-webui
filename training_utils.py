import whisper
import os
import torchaudio
import json
from tqdm import tqdm
import pathlib
import gc
from finetune_speaker_kai import *
import random
from argparse import ArgumentParser
lang2token = {
    'zh': "[ZH]",
    'ja': "[JA]",
    "en": "[EN]",
}
whisper_models = ["tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium", "large-v1",
                                            "large-v2", "large"]
class WhisperTransciber(object):
    def __init__(self,model="medium"):
        self.load_model(model)
    def load_model(self,model):
        self.model = whisper.load_model(model)
    def transcribe_one(self,audio_path):
        model = self.model
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # detect the spoken language
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        lang = max(probs, key=probs.get)
        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)

        # print the recognized text
        print(result.text)
        return lang, result.text
    def transcribe_dataset(self,dataset_folder):
        parent_dir = pathlib.Path(dataset_folder)
        fs = list(parent_dir.iterdir())
        speaker_folders = [f for f in fs if f.is_dir()]
        speaker_names = [f.stem for f in speaker_folders]
        print("Speaker_names:",speaker_names)

        speaker2id = {}
        speaker_annos = []
        for speaker in speaker_names:
            speaker2id[speaker] = len(speaker2id)
        # generate new config
        with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
            hps = json.load(f)

        # modify n_speakers
        # hps['data']["n_speakers"] = 1000 + len(speaker2id)
        hps['data']["n_speakers"] = len(speaker2id)

        # change train/val file path
        hps["data"]['training_files'] = parent_dir.joinpath(hps["data"]['training_files']).__str__()
        hps["data"]['validation_files'] = parent_dir.joinpath(hps["data"]['validation_files']).__str__()

        # clear original speakers
        hps['speakers'] = {}
        # add speaker names
        for speaker in speaker_names:
            hps['speakers'][speaker] = speaker2id[speaker]
        # save modified config
        with open( parent_dir.joinpath("modified_finetune_speaker.json"), 'w', encoding='utf-8') as f:
            json.dump(hps, f, indent=2)


        # resample audios
        for speaker_folder in speaker_folders:
            speaker = speaker_folder.stem
            print(f"Processing speaker:{speaker}")
            speaker_files = list(speaker_folder.iterdir())
            '''
            see how many of them have been processed, pick out the ones that are not processed and only process these guys
            '''
            # processed_files = [f.name for f in speaker_files if f.name.startswith("processed_")]
            # print(f"Found {len(processed_files)} processed voice lines")
            # raw_files = ["processed_"+f for f in speaker_files if not f.name.startswith("processed_")]
            # unprocessed_files = set(raw_files).difference(processed_files)
            # to_process = [ f.replace("processed_",'') for f in unprocessed_files ]
            # to_process_files = [speaker_folder.joinpath(f) for f in to_process]
            #
            # print(f"{len(to_process_files)} to process")
            for i, wavfile in enumerate(tqdm(speaker_files)):
                # try to load file as audio
                if wavfile.name.startswith("processed_"):
                    continue
                try:
                    # wav, sr = torchaudio.load(parent_dir + speaker + "/" + wavfile, frame_offset=0, num_frames=-1, normalize=True,
                    #                           channels_first=True)
                    wav, sr = torchaudio.load(str(wavfile.resolve()) , frame_offset=0, num_frames=-1, normalize=True,
                                              channels_first=True)
                    wav = wav.mean(dim=0).unsqueeze(0)
                    if sr != 22050:
                        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)(wav)
                    if wav.shape[1] / sr > 20:
                        print(f"{wavfile} too long, ignoring\n")
                    save_path = str(speaker_folder.joinpath(f"processed_{i}.wav").resolve())
                    torchaudio.save(save_path, wav, 22050, channels_first=True)
                    # transcribe text
                    lang, text = self.transcribe_one(save_path)
                    if lang not in ['zh', 'en', 'ja']:
                        print(f"{lang} not supported, ignoring\n")
                    text = lang2token[lang] + text + lang2token[lang] + "\n"
                    # print(save_path + "|" + str(speaker2id[speaker]) + "|" + text)
                    speaker_annos.append(save_path + "|" + str(speaker2id[speaker]) + "|" + text)
                except Exception as e:
                    print(e)
                    print(f"Failed to load {wavfile.resolve()}")
                    continue
        # clean annotation
        import text
        cleaned_speaker_annos = []
        for i, line in enumerate(speaker_annos):
            path, sid, txt = line.split("|")
            if len(txt) > 100:
                continue
            cleaned_text = text._clean_text(txt, ["cjke_cleaners2"])
            cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
            cleaned_speaker_annos.append(path + "|" + sid + "|" + cleaned_text)
        with open(parent_dir.joinpath("annotations.txt"), 'w', encoding='utf-8') as f:
            for line in cleaned_speaker_annos:
                f.write(line)
        print(speaker_annos)
        print(f"Found {len(cleaned_speaker_annos)} lines")
        print(f"Finished transcribing {dataset_folder}")
    def annotations_train_val_split(self,dataset_folder,train=1.0,val=1.0):
        dataset_dir = dataset_folder
        if os.path.exists(os.path.join(dataset_dir, "annotations.txt")):
            with open(os.path.join(dataset_dir, "annotations.txt"), 'r', encoding='utf-8') as f:
                custom_character_anno = f.readlines()
            if len(custom_character_anno):
                num_character_voices = len(custom_character_anno)
                final_annos = custom_character_anno
        # save annotation file
        with open(os.path.join(dataset_dir, "final_annotation_train.txt"), 'w', encoding='utf-8') as f:
            for line in final_annos:
                f.write(line)
        # save annotation file for validation
        with open(os.path.join(dataset_dir, "final_annotation_val.txt"), 'w', encoding='utf-8') as f:
            if os.path.exists(os.path.join(dataset_dir, "annotations.txt")):
                for line in custom_character_anno:
                    f.write(line)
    def hibernate(self):
        del self.model

