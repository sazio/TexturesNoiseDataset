import random
import torch
import torch.nn.functional as F
from torchvision import transforms
import PIL 
import os 


#random.seed(2022)


class TexturesNoiseDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform Textures and Noise augmentation for V2 like behavior
    """
    def __init__(self, dataset, preprocess, path_textures, path_noise):
        self.dataset = dataset
        self.preprocess = preprocess
        #self.alpha = alpha
        self.path_textures = path_textures
        self.path_noise = path_noise

    def __getitem__(self, i):
        x, y = self.dataset[i]
        
        
        return (self.preprocess(x), 
                Textures(x,  self.preprocess, self.path_textures),
                Noise(x, self.preprocess, self.path_noise)), y

    def __len__(self):
        return len(self.dataset)

## Blend textures and images
def Textures(x_orig, preprocess, path_textures):
    # alpha : blending constant

    #x_temp = x_orig # back up for skip connection
    alpha = random.uniform(0.005, 0.5)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
    #x_aug = torch.zeros_like(preprocess(x_orig))
    
    
    pil_text = PIL.Image.open(path_textures + random.choice( os.listdir(path_textures)))
    tensor_text = normalize(transforms.ToTensor()(pil_text))
    
    
    blend_tensor_text = (1-alpha)*preprocess(x_orig) + alpha*tensor_text

    return blend_tensor_text


## Blend textures and images
def Noise(x_orig, preprocess, path_noise):
    # alpha : blending constant
    alpha = random.uniform(0.05, 0.5)
    #x_temp = x_orig # back up for skip connection

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
    #x_aug = torch.zeros_like(preprocess(x_orig))
    
    
    pil_text = PIL.Image.open(path_noise + random.choice( os.listdir(path_noise)))
    tensor_text = normalize(transforms.ToTensor()(pil_text))
    
    
    blend_tensor_noise = (1-alpha)*preprocess(x_orig) + alpha*tensor_text

    return blend_tensor_noise