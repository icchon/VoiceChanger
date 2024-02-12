
import os
import sys
import numpy as np
import librosa
import re
import pyopenjtalk
import soundfile as sf
import random
from scipy.io import wavfile
import IPython
from IPython.display import Audio
from tqdm import tqdm
import re
from tacotron.frontend.openjtalk import text_to_sequence, numeric_feature_by_regex, pp_symbols, sequence_to_text
from scipy.stats import zscore
from util import pad_1d, pad_2d
import torch
import torch.utils.data as data
from tacotron import Tacotron2 
from torch import optim
from util import make_non_pad_mask
from torch import nn

def ensure_divisible_by(feats, N):
        if N == 1:
            return feats
        mod = len(feats) % N
        if mod != 0:
            feats = feats[: len(feats) - mod]
        return feats

def collate_fn_tacotron(batch, reduction_factor=1):
    xs = [x[0] for x in batch]
    ys = [ensure_divisible_by(x[1], reduction_factor) for x in batch]
    in_lens = [len(x) for x in xs]
    out_lens = [len(y) for y in ys]
    in_max_len = max(in_lens)
    out_max_len = max(out_lens)
    x_batch = torch.stack([torch.from_numpy(pad_1d(x, in_max_len)) for x in xs])
    y_batch = torch.stack([torch.from_numpy(pad_2d(y, out_max_len)) for y in ys])
    in_lens = torch.tensor(in_lens, dtype=torch.long)
    out_lens = torch.tensor(out_lens, dtype=torch.long)
    stop_flags = torch.zeros(y_batch.shape[0], y_batch.shape[1])
    for idx, out_len in enumerate(out_lens):
        stop_flags[idx, out_len - 1 :] = 1.0
    return x_batch, in_lens, y_batch, out_lens, stop_flags

class Dataset(data.Dataset):
        def __init__(self, X, y):
            self.uids = X
            self.streams = y

        def __len__(self):
            return len(self.uids)

        def __getitem__(self, idx):
            uid, stream = self.uids[idx], self.streams[idx]
            return uid, stream

def main():
    wavfilepath_text = []
    text_path = "./transcript_utf8.txt"
    text_processed_path = "./text_processed.txt"

    n_fft = 2048
    frame_shift = 240
    sr = 16000
    n_mels = 80
    melfb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spec_list = []

    with open(text_path, encoding="utf-8") as f:
        for line in f:
            wav_id, text = line.split(":")
            wav_file_path = os.path.join("./basic5000", f"{wav_id}.wav")
            _sr, x = wavfile.read(wav_file_path)
            x = x.astype(np.float64)
            x = librosa.resample(x, orig_sr=_sr, target_sr=sr)
            X = np.abs(librosa.stft(x.astype(np.float32), n_fft=n_fft, hop_length=frame_shift))
            out_feats = np.dot(melfb, X).T
            out_feats = zscore(out_feats, axis=None)
            mel_spec_list.append(out_feats)
    

    embeded_phoneme_list = []

    with open(text_path, encoding="utf-8") as f:
        for line in f:
            wav_id, text = line.split(":")
            labels = pyopenjtalk.extract_fullcontext(text)
            PP = pp_symbols(labels)
            in_feats = np.array(text_to_sequence(PP), dtype=np.int64)
            embeded_phoneme_list.append(np.array(in_feats))

    X = mel_spec_list
    y = embeded_phoneme_list
    dataset = Dataset(y[:-10], X[:-10])
    test_dataset = Dataset(y[-10:], X[-10:])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=collate_fn_tacotron, num_workers=0)

    model = Tacotron2()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=100000)


    for in_feats, in_lens, out_feats, out_lens, stop_flags in tqdm(dataloader):
        in_lens, indices = torch.sort(in_lens, dim=0, descending=True)
        in_feats, out_feats, out_lens = in_feats[indices], out_feats[indices], out_lens[indices]

        outs, outs_fine, logits, _ = model(in_feats, in_lens, out_feats)

        mask = make_non_pad_mask(out_lens).unsqueeze(-1)
        out_feats = out_feats.masked_select(mask)
        outs = outs.masked_select(mask)
        outs_fine = outs_fine.masked_select(mask)
        stop_flags = stop_flags.masked_select(mask.squeeze(-1))
        logits = logits.masked_select(mask.squeeze(-1))

        decoder_out_loss = nn.MSELoss()(outs, out_feats)
        postnet_out_loss = nn.MSELoss()(outs_fine, out_feats)
        stop_token_loss = nn.BCEWithLogitsLoss()(logits, stop_flags)
    
        loss = decoder_out_loss + postnet_out_loss + stop_token_loss

        print(f"decoder_out_loss: {decoder_out_loss:.2f}, postnet_out_loss: {postnet_out_loss:.2f}, stop_token_loss: {stop_token_loss:.2f}")
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    PATH = 'my_model.pth'
    torch.save(model.state_dict(), PATH)




if __name__ == "__main__":
    main()
