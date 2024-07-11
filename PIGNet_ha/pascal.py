from __future__ import print_function
import cv2
import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import scipy.ndimage as ndi







class VOCSegmentation(data.Dataset):
  CLASSES = [
      'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
      'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
      'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
      'tv/monitor'
  ]

  def __init__(self, root, train=True, transform=None, target_transform=None, download=False, crop_size=None,process=None,process_value=None,overlap_percentage=None):
    self.root = root
    _voc_root = os.path.join(self.root, 'VOC2012')
    _list_dir = os.path.join(_voc_root, 'list')
    self.transform = transform
    self.target_transform = target_transform
    self.train = train
    self.crop_size = crop_size
    self.process = process
    self.process_value = process_value
    self.overlap_percentage = overlap_percentage

    if download:
      self.download()

    if self.train:
      _list_f = os.path.join(_list_dir, 'train_aug.txt')
    else:
      _list_f = os.path.join(_list_dir, 'val.txt')

    if self.process == 'overlap':
      print("!! process model")
    elif self.process == 'zoom':
      print("!! zoom_in model")


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


    # add image process for test
    if self.process == 'zoom':
      _img, _target = self.zoom_center(_img, _target, self.process_value)

    elif self.process == 'overlap' and index < len(self.images) - 1:
      next_img=Image.open(self.images[index+1]).convert('RGB')
      next_target=Image.open(self.masks[index+1])
      _img, _target = self.overlap(_img, _target,next_img,next_target,self.overlap_percentage)

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

  def overlap(self, image, mask, next_image, next_mask, overlap_percentage=0.5):
    # 이미지의 크기 가져오기
    width1, height1 = image.size
    width2, height2 = next_image.size

    # 겹칠 영역 계산
    overlap_width = int(width2 * overlap_percentage)
    overlap_height = int(height2 * overlap_percentage)

    # 큰 이미지를 생성하고 첫 번째 이미지를 왼쪽 위에 배치
    result_image = Image.new('RGB', (width1+width2, height1+height2))
    result_mask = Image.new('L', (width1+width2, height1+height2))

    result_image.paste(image, (0, 0))
    result_mask.paste(mask, (0, 0))

    # 두 번째 이미지를 오른쪽 아래에 배치
    offset_x = width1 - overlap_width
    offset_y = height1 - overlap_height

    result_image.paste(next_image, (offset_x, offset_y))
    result_mask.paste(next_mask, (offset_x, offset_y))

    # 결과 이미지와 마스크를 원래 이미지 크기로 줄이기
    result_image = result_image.crop((0,0,width1, height1))
    result_mask = result_mask.crop((0,0,width1, height1))

    return result_image, result_mask

  def zoom_center(self, image, mask, zoom_factor):
    """
    Zooms into or out of the image and mask around the center by the given zoom_factor.
    Keeps the image size unchanged.
    """
    width, height = image.size

    if zoom_factor > 1:
      # Zoom in
      new_width = int(width / zoom_factor)
      new_height = int(height / zoom_factor)

      # Center coordinates
      center_x, center_y = width // 2, height // 2

      # Calculate the crop box
      left = max(center_x - new_width // 2, 0)
      right = min(center_x + new_width // 2, width)
      top = max(center_y - new_height // 2, 0)
      bottom = min(center_y + new_height // 2, height)

      # Crop the image and mask, then resize back to the original dimensions
      image = image.crop((left, top, right, bottom)).resize((width, height), Image.Resampling.LANCZOS)
      mask = mask.crop((left, top, right, bottom)).resize((width, height), Image.Resampling.NEAREST)

    elif zoom_factor < 1:
      # Zoom out
      new_width = int(width * zoom_factor)
      new_height = int(height * zoom_factor)

      # Resize the image and mask to the new dimensions
      resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
      resized_mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)

      # Create a new black image and paste the resized image and mask in the center
      new_image = Image.new('RGB', (width, height), (255, 255, 255))
      new_mask = Image.new('L', (width, height), 255)

      new_image.paste(resized_image, ((width - new_width) // 2, (height - new_height) // 2))
      new_mask.paste(resized_mask, ((width - new_width) // 2, (height - new_height) // 2))

      image = new_image
      mask = new_mask

    return image, mask

