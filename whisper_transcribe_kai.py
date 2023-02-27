import whisper
import os
import torchaudio
import json
from tqdm import tqdm
from argparse import ArgumentParser
lang2token = {
    'zh': "[ZH]",
    'ja': "[JA]",
    "en": "[EN]",
}


def transcribe_one(audio_path):
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
if __name__ == "__main__":
    whisper_models = ["tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium", "large-v1",
                      "large-v2", "large"]
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default="custom_character_voice",
                        help=' dataset folder ')
    parser.add_argument('-o', '--output_dir', type=str, default="custom_character_voice",
                        help='output folder')
    parser.add_argument('-m', '--whisper_model',dest="model", type=str, default="medium",
                        help=f'whisper model to choose, options are {whisper_models}')

    args = parser.parse_args()
    model = whisper.load_model(args.model)
    parent_dir = args.input_dir+"/"

    speaker_names = list(os.walk(parent_dir))[0][1]
    speaker2id = {}
    speaker_annos = []
    # resample audios
    for speaker in speaker_names:
        speaker2id[speaker] = 1000 + len(speaker2id)
        print(f"Processing speaker:{speaker}")
        for i, wavfile in enumerate(tqdm(list(os.walk(parent_dir + speaker))[0][2])):
            # try to load file as audio
            if wavfile.startswith("processed_"):
                continue
            try:
                wav, sr = torchaudio.load(parent_dir + speaker + "/" + wavfile, frame_offset=0, num_frames=-1, normalize=True,
                                          channels_first=True)
                wav = wav.mean(dim=0).unsqueeze(0)
                if sr != 22050:
                    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)(wav)
                if wav.shape[1] / sr > 20:
                    print(f"{wavfile} too long, ignoring\n")
                save_path = parent_dir + speaker + "/" + f"processed_{i}.wav"
                torchaudio.save(save_path, wav, 22050, channels_first=True)
                # transcribe text
                lang, text = transcribe_one(save_path)
                if lang not in ['zh', 'en', 'ja']:
                    print(f"{lang} not supported, ignoring\n")
                text = lang2token[lang] + text + lang2token[lang] + "\n"
                speaker_annos.append(save_path + "|" + str(speaker2id[speaker]) + "|" + text)
            except:
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
    with open(f"{parent_dir}annotations.txt", 'w', encoding='utf-8') as f:
        for line in cleaned_speaker_annos:
            f.write(line)
    # generate new config
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)

    # modify n_speakers
    hps['data']["n_speakers"] = 1000 + len(speaker2id)

    # change train/val file path
    hps["data"]['training_files'] = parent_dir+hps["data"]['training_files']
    hps["data"]['validation_files'] = parent_dir + hps["data"]['validation_files']

    # add speaker names
    for speaker in speaker_names:
        hps['speakers'][speaker] = speaker2id[speaker]
    # save modified config
    with open(f"{parent_dir}modified_finetune_speaker.json", 'w', encoding='utf-8') as f:
        json.dump(hps, f, indent=2)
    print("finished")