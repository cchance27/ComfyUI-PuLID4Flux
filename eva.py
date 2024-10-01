import torch
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from eva_clip.factory import create_model_and_transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize
from pulid_utils import colored

class EVAClip:
    def __init__(self, device="cpu", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        
        print(colored(0,0,255, "EVA02"), " Loading EVA-CLIP02")
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        model = model.visual
        self.clip_vision_model = model.to(self.device, dtype=self.dtype)
        
        eva_transform_mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std 
    
    @torch.inference_mode()
    def get_cond_and_hidden(self, face_features_image):
        print(colored(0,0,255, "EVA02"), " Generating Conditioning")

        face_features_image = resize(face_features_image, self.clip_vision_model.image_size, InterpolationMode.NEAREST if torch.backends.mps.is_available() else InterpolationMode.BICUBIC, 1024).to(self.device, dtype=self.dtype)
        face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
        id_cond_vit, id_vit_hidden = self.clip_vision_model(
            face_features_image.to(self.dtype), return_all_features=False, return_hidden=True, shuffle=False
        )
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)
        
        return (id_cond_vit, id_vit_hidden)
    