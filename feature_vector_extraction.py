'''
Extract feature vectors from video frames.
These features come from the Pool5 layers of a ResNet deep
neural network, pre-trained on ImageNet. The algorithm captures
frames directly from video, there is not need for prior frame extraction.

Copyright (C) 2019 Alexandros I. Metsai
alexmetsai@gmail.com

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import os
import torch
from torchvision import transforms, models
import torch.nn as nn
import cv2
from PIL import Image

class Rescale(object):
    """
    Rescale an image to the given size.

    Args:
        output_size : Can be int or tuple. In the case a single integer
        is given, PIL will resize the smallest of the original
        dimensions to this value, and resize the largest dimention 
        such as to keep the same aspect ratio.
    """

    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        """
        Args:
            image (PIL.Image) : PIL.Image object to rescale
        """
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return img


class ResNetPool5(nn.Module):
    # TODO
    def __init__(self, DNN='resnet101'):
        """
        Load pretrained ResNet weights on ImageNet. Return the Pool5
        features as output when called.
        
        Args:
            DNN (string): The DNN architecture. Choose from resnet101, 
            resnet50 or resnet152. ResNet50 and ResNet152 are not yet 
            in the release version of TorchVision, you will have to 
            build from source for these nets to work, or wait for the
            newer versions.
        """
        super().__init__()
        
        if DNN == "resnet101":
            resnet = models.resnet101(pretrained=True)
        elif DNN == "resnet50":
            resnet = models.resnet50(pretrained=True)
        elif DNN == "resnet152":
            resnet = models.resnet152(pretrained=True)
        else:
            print("Error. Network " + DNN + " not supported.")
            exit(1)
        resnet.float()
        
        # Use GPU is possible
        if torch.cuda.is_available():
            resnet.cuda()
        resnet.eval()
        
        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[:-2])
        self.pool5 = module_list[-2]
        
    def forward(self, x):
        res5c = self.conv5(x)
        pool5 = self.pool5(res5c)
        pool5 = pool5.view(pool5.size(0), -1)
        return pool5

# Check torchvision docs about these normalization values applied on ResNet.
# Since it was applied on training data, we should do so as well.
# Note that this transform does not cast the data first to [0,1].
# Should this action be performed? Or does this normalization make
# it uneccessary?
data_normalization = transforms.Compose([
    Rescale(224, 224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])

if __name__=='__main__':
    
    model = ResNetPool5()
    video_folder = "videos/"
    # Extract features for all the videos in the list.
    
    for file in os.listdir(video_folder):
        
        # Empty list to append tensors.
        features_list = []
        
        if file.endswith(".mp4"):
            print("Processing " + file)
            video_capture = cv2.VideoCapture(video_folder + file)
            success, image = video_capture.read()
            i = 1
            while success:
