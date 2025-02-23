import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import time

import torchvision.transforms as transforms

def image_loader(image_name):
    loader = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.ToTensor()])  
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to("cpu", torch.float)

def imshow(tensor, title=None, save=False):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    plt.figure()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    if save:
        timestr = time.strftime("%Y%m%d_%H%M%S")
        image.save(f'./output/{timestr}.png')
        print(f"Output saved at location:: ./output/{timestr}.png")


# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        

    def forward(self, img):
        # normalize ``img``
        return (img - self.vgg_mean) / self.vgg_std
