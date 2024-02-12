
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.io import wavfile
import librosa.display
import pyopenjtalk
from tacotron.frontend.openjtalk import text_to_sequence, numeric_feature_by_regex, pp_symbols
from tacotron import Tacotron2 
import pyaudio
import wave
import struct
import speech_recognition
from tqdm import tqdm
from IPython.display import Audio
from tacotron import tacotron2

sr = 16000

PATH = './my_model.pth'
model_params = torch.load(PATH)
model = Tacotron2()
model.load_state_dict(model_params)

def wav2vec(filename):
    r = speech_recognition.Recognizer()
    with speech_recognition.AudioFile(filename) as source:
        audio_data = r.record(source)
    text = r.recognize_google(audio_data, language="ja-JP")
    print(text)
    labels = pyopenjtalk.extract_fullcontext(text)
    
    PP = pp_symbols(labels)
    in_feat = np.array(text_to_sequence(PP), dtype=np.int64)
    
    return in_feat

def mel2wav(x):
    n_fft = 2048
    frame_shift = 240
    sr = 16000
    gen_wav = librosa.feature.inverse.mel_to_audio(M=x.T, sr=sr, n_fft=n_fft, hop_length=frame_shift)
    return gen_wav

def voice2voice(record_second=2**3):
    CHUNK = 2**16 
    RATE = CHUNK // record_second
    FORMAT = pyaudio.paInt16 
    CHANNELS = 1 

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,   
                      )
  
    in_data = stream.read(CHUNK)
    
    output_path = "./output.wav"
    wf = wave.open(output_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(in_data)
    wf.close()

    filename = "./output.wav"
    in_feat = wav2vec(filename)
    in_feat = torch.tensor(in_feat, dtype=torch.long)
    
    mel = model.inference(in_feat)[0].detach().numpy()
    wav = mel2wav(mel)
    wav = ((wav - wav.mean()) / wav.std())*(2**12)
    
    data_binary = wav.astype('int16').astype(np.int16).tobytes()
    stream.write(data_binary)
    
    return wav


def main():
    Audio(voice2voice(record_second=4), rate=sr)

if __name__ == "__main__":
    main()
