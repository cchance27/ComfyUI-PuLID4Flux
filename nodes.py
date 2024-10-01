import torch
import torchvision.transforms as T
import torch.nn.functional as F
from pulid import PuLID
from eva import EVAClip
from facedet import FaceDetection
from patcher import PULIDModelPatcher
from pulid_utils import colored

class PULIDLoadFaceDetectionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "device": (["cuda", "mps", "cpu"],)}
        }
    
    RETURN_TYPES = ("facedetection", )
    FUNCTION = "load_facedetection"
    CATEGORY = "PuLID/FLUX"

    def load_facedetection(self, device="cpu"):
        face = FaceDetection(device=device)
        
        return (face, )   
 

class PULIDEvaClipNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "device": (["cuda", "mps", "cpu"],)}
        }
    
    RETURN_TYPES = ("evaclip", )
    FUNCTION = "load_evaclip"
    CATEGORY = "PuLID/FLUX"

    def load_evaclip(self, device="cpu"):
        eva = EVAClip(device=device)
        
        return (eva, ) 
    
 
class PULIDFluxPatcher:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                
                "model": ("MODEL",),
                "evaclip": ("evaclip",),
                "facedetection": ("facedetection",),
                "id_image": ("IMAGE",),
                'id_weight': ('FLOAT', {'min': 0.0, 'max': 2.0, 'step': 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE")
    FUNCTION = "patch_model"
    CATEGORY = "PuLID/FLUX"

    def patch_model(self, model, facedetection, evaclip, id_image, id_weight):
        print(colored(255, 0, 0, 'PuLID Patcher') + ' Initialize PuLID')
        pulid = PuLID(model.load_device, dtype=torch.float16, id_weight=id_weight)
        print(colored(255, 0, 0, 'PuLID Patcher') + ' Loading PuLID Model')
        pulid.load_checkpoint()
        
        print(colored(255, 0, 0, 'PuLID Patcher') + ' Cloning Model into Patcher')
        patcher = PULIDModelPatcher.clone(model, pulid=pulid)
        print(colored(255, 0, 0, 'PuLID Patcher') + ' Done Cloning')
        
        face_feature_tensor, ante_embed = facedetection.get_face_and_ante_embedding(id_image)
        eva_cond, eva_hidden = evaclip.get_cond_and_hidden(face_feature_tensor)
        pulid.generate_pulid_embedding(ante_embed, eva_cond, eva_hidden)
        
        patcher.patch_model(pulid)
        
        return (patcher, facedetection.debug_img_list) 
 
 
NODE_CLASS_MAPPINGS = {
    "PULIDLoadFaceDetection": PULIDLoadFaceDetectionNode,
    "PULIDEVAClip": PULIDEvaClipNode,
    "PULIDFluxPatcher": PULIDFluxPatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PULIDLoadFaceDetection": "(Down)load Face Detection",
    "PULIDFluxPatcher": "(Down)load & Apply PuLID", 
    "PULIDEVAClip": "(Down)load Eva Clip",
}