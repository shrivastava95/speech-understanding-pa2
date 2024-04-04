from tqdm import tqdm
import soundfile as sf
import torch
import fire
import torch.nn.functional as F
from torchaudio.transforms import Resample
from models.ecapa_tdnn import ECAPA_TDNN_SMALL

MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_base_plus", "wavlm_large"]
MODEL_LIST = ['hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_large"]

def init_model(model_name, checkpoint=None):
    if model_name == 'unispeech_sat':
        config_path = 'config/unispeech_sat.th'
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='unispeech_sat_large', config_path=config_path)
    elif model_name == 'wavlm_base_plus':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=config_path)
    elif model_name == 'wavlm_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
    elif model_name == 'hubert_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k', config_path=config_path)
    elif model_name == 'wav2vec2_xlsr':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='xlsr_53', config_path=config_path)
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model


def verification(model_name,  wav1, wav2, use_gpu=True, checkpoint=None):

    assert model_name in MODEL_LIST, 'The model_name should be in {}'.format(MODEL_LIST)
    model = init_model(model_name, checkpoint)

    wav1, sr1 = sf.read(wav1)
    wav2, sr2 = sf.read(wav2)

    wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
    wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
    resample1 = Resample(orig_freq=sr1, new_freq=16000)
    resample2 = Resample(orig_freq=sr2, new_freq=16000)
    wav1 = resample1(wav1)
    wav2 = resample2(wav2)

    if use_gpu:
        model = model.cuda()
        wav1 = wav1.cuda()
        wav2 = wav2.cuda()

    model.eval()
    with torch.no_grad():
        emb1 = model(wav1)
        emb2 = model(wav2)

    sim = F.cosine_similarity(emb1, emb2)
    print("The similarity score between two audios is {:.4f} (-1.0, 1.0).".format(sim[0].item()))


def read_soundfile_from_path(path):
    wav, sr = sf.read(path)
    wav = torch.from_numpy(wav).unsqueeze(0).float()
    sampling_rate = 16000
    resample = Resample(orig_freq=sr, new_freq=sampling_rate)
    return resample(wav)

def verification_multi(model_name, list_wav12, use_gpu=True, checkpoint=None):
    assert model_name in MODEL_LIST, "The model_name should be in {}".format(MODEL_LIST)
    model = init_model(model_name, checkpoint)
    model.eval()

    if use_gpu:
        model = model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    sim_scores = []
    
    with torch.no_grad():
        for path1, path2 in tqdm(list_wav12):
            wav1 = read_soundfile_from_path(path1).to(device)
            wav2 = read_soundfile_from_path(path2).to(device)
            emb1 = model(wav1)
            emb2 = model(wav2)
            sim = F.cosine_similarity(emb1, emb2)
            sim_scores.append(sim)
    
    return sim_scores
    


if __name__ == "__main__":
    fire.Fire(verification)

