# Extract feature vectors from video frames.
# These features come from the Pool5 layers of a ResNet deep
# neural network, pre-trained on ImageNet. The algorithm captures
# frames directly from video, there is not need for prior frame extraction.

# Copyright (C) 2019 Alexandros I. Metsai
# alexmetsai@gmail.com

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import torch
from torchvision import transforms, models
import torch.nn as nn
import cv2
from PIL import Image

class Rescale(object):
  # TODO
