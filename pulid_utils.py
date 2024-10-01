import cv2
import torch

def get_onnx_providers():
    providers = []
    
    if torch.cuda.is_available():
        providers.append('CUDAExecutionProvider')
    if torch.backends.mps.is_available():
        providers.append('CoreMLExecutionProvider')
    providers.append('CPUExecutionProvider')
    
    print("Using providers:", providers)
    return providers

def colored(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def tensor_to_image(tensor):
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [2, 1, 0]].numpy()
    print(image.shape)
    return image

def image_to_tensor(image):
    tensor = torch.clamp(torch.from_numpy(image).float() / 255., 0, 1)
    tensor = tensor[..., [2, 1, 0]]
    return tensor

def resize_numpy_image_long(image, resize_long_edge=768):
    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image
