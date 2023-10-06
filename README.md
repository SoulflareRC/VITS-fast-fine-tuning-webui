## VITS-fast-fine-tuning-webui<br>
[demo](/demo/extract_voiceline.png)
This repo is a fork of Plachtaa's [VITS-fast-fine-tuning repo](https://github.com/Plachtaa/VITS-fast-fine-tuning). For details about the project, please refer to the original repo. 
## New changes
- Supports previewing dataset
- Supports editing dataset
- Supports extract voicelines using subtitles file and slice audios by silence
- Supports resume training a trained model
- No longer requires sample audios
- Custome characters only
- More flexible config
## Installation
1. Clone this repo, set up virtual environment, and install requirements.<br>
```
git clone https://github.com/SoulflareRC/VITS-fast-fine-tuning-webui.git 
cd VITS-fast-fine-tuning-webui 
python -m venv venv
venv\Script\activate
pip install -r requirements.txt
```
2. Create a folder ```pretrained_models```, and download the pretrained models from the original repo <br> 
https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/D_trilingual.pth
https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/G_trilingual.pth
Put the models in the folder. 
3. Build ```monotonic_align```
```
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
```
4. Launch the web ui and enjoy!
```
python gradio_interface.py
```
