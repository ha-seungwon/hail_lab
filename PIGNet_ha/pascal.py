from __future__ import print_function

import torch.utils.data as data
import os
from PIL import Image
from utils import preprocess


class VOCSegmentation(data.Dataset):
  CLASSES = [
      'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
      'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
      'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
      'tv/monitor'
  ]

  def __init__(self, root, train=True, transform=None, target_transform=None, download=False, crop_size=None,process=None):
    self.root = root
    _voc_root = os.path.join(self.root, 'VOC2012')
    _list_dir = os.path.join(_voc_root, 'list')
    self.transform = transform
    self.target_transform = target_transform
    self.train = train
    self.crop_size = crop_size
    self.process = process

    if download:
      self.download()

    if self.train:
      _list_f = os.path.join(_list_dir, 'train_aug.txt')
    else:
      _list_f = os.path.join(_list_dir, 'val.txt')

    if self.process == 'overlap':
      print("!! process model")
      _list_f = os.path.join(_list_dir, 'process_val.txt')
    elif self.process == 'zoom_in':
      print("!! zoom_in model")
      _list_f = os.path.join(_list_dir, 'zoom_in.txt')
    elif self.process == 'zoom_out':
      print("!! zoom_out model")
      _list_f = os.path.join(_list_dir, 'zoom_out.txt')


    self.images = []
    self.masks = []
    with open(_list_f, 'r') as lines:
      for line in lines:
        _image = _voc_root + line.split()[0]
        _mask = _voc_root + line.split()[1]
        assert os.path.isfile(_image)
        assert os.path.isfile(_mask)
        self.images.append(_image)
        self.masks.append(_mask)

  def __getitem__(self, index):
    _img = Image.open(self.images[index]).convert('RGB')
    _target = Image.open(self.masks[index])
    if self.process != None:
      _target= _target.convert('L')


    _img, _target = preprocess(_img, _target,
                               flip=True if self.train else False,
                               scale=(0.5, 2.0) if self.train else None,
                               crop=(self.crop_size, self.crop_size))

    if self.transform is not None:
      _img = self.transform(_img)

    #print("self.target_transform ",self.target_transform )
    if self.target_transform is not None:
      _target = _target.unsqueeze(0)
      _target = self.target_transform(_target)

    return _img, _target

  def __len__(self):
    return len(self.images)

  def download(self):
    raise NotImplementedError('Automatic download not yet implemented.')

