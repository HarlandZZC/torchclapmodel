dependencies = ['torch', 'numpy', 'resampy', 'soundfile']

import laion_clap
import torch
from packaging import version
import transformers

def clapmodel(weights_name, device):
    model_urls = {
    'music_speech_audioset_epoch_15_esc_89.98.pt': 'https://huggingface.co/lukewys/laion_clap/resolve/main/'
            'music_speech_audioset_epoch_15_esc_89.98.pt',
    'music_audioset_epoch_15_esc_90.14.pt': 'https://huggingface.co/lukewys/laion_clap/resolve/main/'
           'music_audioset_epoch_15_esc_90.14.pt',
    'music_speech_epoch_15_esc_89.25.pt': 'https://huggingface.co/lukewys/laion_clap/resolve/main/'
            'music_speech_epoch_15_esc_89.25.pt',
    '630k-audioset-fusion-best.pt': 'https://huggingface.co/lukewys/laion_clap/resolve/main/'
            '630k-audioset-fusion-best.pt'}
    
    def load_state_dict_from_url(url: str, map_location="cpu", skip_params=True, progress=True):
        checkpoint = torch.hub.load_state_dict_from_url(url, progress=progress, map_location=map_location)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        if skip_params:
            if next(iter(state_dict.items()))[0].startswith("module"):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
        if version.parse(transformers.__version__) >= version.parse("4.31.0"): 
            del state_dict["text_branch.embeddings.position_ids"]
        return state_dict
    
    if weights_name == "music_speech_audioset_epoch_15_esc_89.98.pt":
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device=device)
        pkg = load_state_dict_from_url(model_urls['music_speech_audioset_epoch_15_esc_89.98.pt'], progress=True)
    elif weights_name == "music_audioset_epoch_15_esc_90.14.pt":
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device=device)
        pkg = load_state_dict_from_url(model_urls['music_audioset_epoch_15_esc_90.14.pt'], progress=True)
    elif weights_name == "music_speech_epoch_15_esc_89.25.pt":
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base',  device=device)
        pkg = load_state_dict_from_url(model_urls['music_speech_epoch_15_esc_89.25.pt'], progress=True)
    elif weights_name == "630k-audioset-fusion-best.pt":
        model = laion_clap.CLAP_Module(enable_fusion=True,  device=device)
        pkg = load_state_dict_from_url(model_urls['630k-audioset-fusion-best.pt'], progress=True)
    else:
        raise ValueError('clap_model not implemented')

    pkg.pop('text_branch.embeddings.position_ids', None)
    model.model.load_state_dict(pkg)

    return model