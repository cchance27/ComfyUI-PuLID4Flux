import os
import torch
import numpy as np
from torchvision.transforms.functional import normalize
import insightface
from insightface.app import FaceAnalysis
from huggingface_hub import snapshot_download
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from pulid_utils import colored, get_onnx_providers
import folder_paths
from pulid_utils import resize_numpy_image_long, tensor_to_image, image_to_tensor

class FaceDetection:
    def __init__(self, device="cpu", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        
        print(colored(0,255,0, "RetinaFace") + " Loading RetinaFace")
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
            model_rootpath = os.path.join(folder_paths.models_dir, "facedetection")
        ) # Sets up face_helper.face_det to retinaface_resnet50 to get BoundingBox and features, and downloads the necessary models
        
        print(colored(0,255,0, "RetinaFace") + " Switch to FaceRestore Parser (bisenet)") #bisenet for speed, parsenet for accuracy
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.device, model_rootpath="facedetection")
        
        print(colored(0,255,0, "AntelopeV2"), " Loading Model") # Balanced facial detection
        snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')
        providers = get_onnx_providers()
        insightface_dir = os.path.join(folder_paths.models_dir, "insightface")
        self.ante = FaceAnalysis(name='antelopev2', root=insightface_dir, providers=providers)
        self.ante.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx', root=insightface_dir, providers=providers)
        self.handler_ante.prepare(ctx_id=0)
        
    def to(self, device="cpu"):
        self.face_helper.face_det.to(device)
        self.face_helper.face_parse.to(device)
        self.handler_ante.to(device)

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x
    
    def reset(self):
        import gc
        
        self.face_helper.clean_all()
        self.debug_img_list = []
        if torch.mps.is_available():
            torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    @torch.inference_mode()
    def get_face_and_ante_embedding(self, image_tensor):
        self.face_helper.clean_all()
        self.debug_img_list = []
        image_img = tensor_to_image(image_tensor)
        image_img = np.squeeze(image_img, axis=0) #Squeeze out the batch
        image_img = resize_numpy_image_long(image_img, 1024) # Limit long size to 1024
        self.debug_img_list.append(image_to_tensor(image_img))
        
        # Get face bounding box with AntelopeV2
        print(colored(0,255,0, "AntelopeV2") + " Processing Image with AntelopeV2")
        face_info = self.ante.get(image_img)
        if len(face_info) > 0:
            print(colored(0,255,0,"AntelopeV2") + " Found a Face / Embedding")
            face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]  # Get the biggest face only
            id_ante_embedding = face_info['embedding']
        else:
            id_ante_embedding = None

        print(colored(0,255,0,"RetinaFace") + " Aligning Face with RetinaFace")
        # Use Retina face to get the face and align it.
        self.face_helper.read_image(image_img)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError('RetinaFace failed to detect a face')
        
        align_face = self.face_helper.cropped_faces[0]
        self.debug_img_list.append(image_to_tensor(self.face_helper.cropped_faces[0]))

        self.debug_img_list.append(image_to_tensor(align_face))

        # AntelopeV2 didn't find the face
        if id_ante_embedding is None:
            print(colored(0,255,0,"AntelopeV2") + " failed to find face so using face from RetinaFace")
            id_ante_embedding = self.handler_ante.get_feat(align_face)

        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device, self.dtype)
        if id_ante_embedding.ndim == 1:
            print(colored(0,255,0,"AntelopeV2") + " using generated embedding from RetinaFace Image")
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        # Parse the face away from any background
        print(colored(0,255,0,"Face Detection") + " Isolating facial features")
        input_tensor = image_to_tensor(align_face).unsqueeze(0) 
        input_tensor = input_tensor.permute(0,3,1,2).to(self.device) # Move to the device
        parsing_out = self.face_helper.face_parse(normalize(input_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out1 = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out1 == i for i in bg_label).bool().repeat(1, 3, 1, 1)
        white_image = torch.ones_like(input_tensor)
        #self.debug_img_list.append(white_image.squeeze(0).permute(1,2,0))             
        gray_image = self.to_gray(input_tensor)
        self.debug_img_list.append(gray_image.squeeze(0).permute(1,2,0))   
        face_feature_tensor = torch.where(bg, white_image, gray_image)
        self.debug_img_list.append(face_feature_tensor.squeeze(0).permute(1,2,0))        

        return (face_feature_tensor, id_ante_embedding)
