from pulid_utils import colored 
import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import torchvision.transforms as T
import torch.nn.functional as F
import folder_paths
from encoders_flux import IDFormer, PerceiverAttentionCA

class PuLID(nn.Module):
    def __init__(self, device="cpu", dtype=torch.float16, model_name='pulid_flux_v0.9.0.safetensors', id_weight=1.0):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.double_interval = 2
        self.single_interval = 4
        self.model_name = model_name
        self.id_weight=id_weight
        
        num_ca = 19 // self.double_interval + 38 // self.single_interval
        if 19 % self.double_interval != 0:
            num_ca += 1
        if 38 % self.single_interval != 0:
            num_ca += 1
            
        self.pulid_encoder = IDFormer().to(self.device, self.dtype)
        self.pulid_ca = nn.ModuleList([
            PerceiverAttentionCA().to(self.device, self.dtype) for _ in range(num_ca)
        ])    
    
    @torch.inference_mode()
    def generate_pulid_embedding(self, ante_embed, eva_cond, eva_hidden):
        id_cond = torch.cat([ante_embed, eva_cond], dim=-1)
        id_embedding = self.pulid_encoder(id_cond, eva_hidden)

        id_uncond = torch.zeros_like(id_cond)
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(eva_hidden)):
            id_vit_hidden_uncond.append(torch.zeros_like(eva_hidden[layer_idx]))

        uncond_id_embedding = self.pulid_encoder(id_uncond, id_vit_hidden_uncond)
        
        self.id_embedding = id_embedding
        self.uncond_id_embedding = uncond_id_embedding

    def load_checkpoint(self, model_name='pulid_flux_v0.9.0.safetensors'):
        print(colored(255,0,0,"PuLID") + ' Loading ' + model_name)
        from safetensors.torch import load_file
        
        download_path = os.path.join(folder_paths.models_dir, "pulid")
        model_path = os.path.join(download_path, model_name)
        if not os.path.exists(model_path):
            print(colored(255,0,0,"PuLID") + f" Downloading PULID For Flux model to: {model_path}")
            hf_hub_download('guozinan/PuLID', model_name, local_dir=download_path)
        
        state_dict = load_file(model_path)
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1:]
            state_dict_dict[module][new_k] = v

        #Load PuLID into the current module from the safetensor.
        for module in state_dict_dict:
            print(colored(255,0,0,"PuLID") + f' Loading {module}')
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)
            
        print(colored(255,0,0,"PuLID") + ' Done Loading')
        
        del state_dict
        del state_dict_dict