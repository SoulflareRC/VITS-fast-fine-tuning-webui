import numpy as np
import torch
from torch import no_grad, LongTensor
import commons
import utils
from models_infer import SynthesizerTrn
from pydub import AudioSegment
from text import text_to_sequence, _clean_text
device = "cuda:0" if torch.cuda.is_available() else "cpu"
language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}
lang = ['日本語', '简体中文', 'English', 'Mix']
def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

class TTSGenerator(object):
    def __init__(self,model_path,config_path):
        self.load_model(model_path,config_path)
    def load_model(self,model_path,config_path):
        print(f"Loading model from {model_path},{config_path}")
        hps = utils.get_hparams_from_file(config_path)
        net_g = SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model).to(device)
        _ = net_g.eval()
        _ = utils.load_checkpoint(model_path, net_g, None)
        speaker_ids = hps.speakers
        speakers = list(hps.speakers.keys())
        self.hps = hps
        self.speaker_ids = speaker_ids
        self.speakers = speakers
        self.model = net_g
        self.speed = 1.0
        print('Successfully loaded model')
    def tts_infer(self,text,speaker,language)->AudioSegment:
        hps = self.hps
        if language is not None:
            text = language_marks[language] + text + language_marks[language]
        speaker_id = self.speaker_ids[speaker]
        stn_tst = get_text(text, hps, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = self.model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / self.speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
        seg = AudioSegment(data=audio.tobytes(), frame_rate=hps.data.sampling_rate, sample_width=audio.dtype.itemsize,
                           channels=1)
        return seg
    def hibernate(self):
        del self.model
if __name__ == "__main__":
    pass
# tts = TTSGenerator('models/kyouma.pth','models/kyouma.json')
#
# audio = tts.tts_infer("こんにちは","kyouma","日本語")
# audio.export("testp.mp3",format="mp3")
# write_audio(audio,"test.wav")
# print(sampling_rate)


# from pydub.playback import play
# play(seg)

# write_audio(audio,"test.wav")

# def create_tts_fn(model, hps, speaker_ids):
#     def tts_fn(text, speaker, language, speed):
#         if language is not None:
#             text = language_marks[language] + text + language_marks[language]
#         speaker_id = speaker_ids[speaker]
#         stn_tst = get_text(text, hps, False)
#         with no_grad():
#             x_tst = stn_tst.unsqueeze(0).to(device)
#             x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
#             sid = LongTensor([speaker_id]).to(device)
#             audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
#                                 length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
#         del stn_tst, x_tst, x_tst_lengths, sid
#         return "Success", (hps.data.sampling_rate, audio)
#
#     return tts_fn
#     args = parser.parse_args()
#     hps = utils.get_hparams_from_file(args.config_dir)
#
#
#
#
#     tts_fn = create_tts_fn(net_g, hps, speaker_ids)