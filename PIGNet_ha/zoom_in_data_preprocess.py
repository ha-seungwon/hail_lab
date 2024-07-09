from torchvision import transforms
import torchvision
from torchvision import transforms

# Define the desired size
# desired_size = 128
#
# # Calculate the padding needed for zoom out
# padding_zoom_out = (desired_size - 32) // 2
#
# # Define the transformation pipeline for zoom in and zoom out
# transform_zoom_in = transforms.Compose([
#     transforms.Resize((int(desired_size * 2), int(desired_size * 2))),  # Slight enlargement
#     transforms.CenterCrop(desired_size)  # Crop to desired size
# ])
#
# transform_zoom_out = transforms.Compose([
#     transforms.Pad(padding_zoom_out, fill=0, padding_mode='constant'),  # Pad the image with zeros
#     transforms.Resize((desired_size, desired_size))  # Resize to desired size
# ])
#
# # Rest of your code remains the same, you can apply the desired transformation based on your requirement
# transform=transform_zoom_out
#
# # CIFAR-100 데이터셋 로드
# dataset = torchvision.datasets.CIFAR10(root='C:/Users/hail/Desktop/ha/model', train=True, download=True,transform=transform)
# valid_dataset = torchvision.datasets.CIFAR10(root='C:/Users/hail/Desktop/ha/model', train=False, download=True, transform=transform)
# dataset.CLASSES = ['plane', 'car', 'bird', 'cat',
#                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#
# print(dataset[0][0].size)
# print(dataset[0][0].show())


from pascal import VOCSegmentation

# Define the desired size
desired_size = 513

# Calculate the padding needed for zoom out
padding_zoom_out = (desired_size - 256) // 2

# Define the transformation pipeline for zoom in and zoom out
transform_zoom_in = transforms.Compose([
    transforms.Resize((int(desired_size * 2), int(desired_size * 2))),  # Slight enlargement
    transforms.CenterCrop(desired_size)  # Crop to desired size
])

transform_zoom_out = transforms.Compose([
    transforms.Pad(padding_zoom_out, fill=0, padding_mode='constant'),  # Pad the image with zeros
    transforms.Resize((desired_size, desired_size))  # Resize to desired size
])
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False,
                    help='training mode')
parser.add_argument('--exp', type=str,default="bn_lr7e-3",
                    help='name of experiment')
parser.add_argument('--gpu', type=int, default=0,
                    help='test time gpu device id')
parser.add_argument('--backbone', type=str, default='resnet50',
                    help='resnet50')
parser.add_argument('--dataset', type=str, default='pascal',
                    help='pascal or cityscapes')
parser.add_argument('--groups', type=int, default=None,
                    help='num of groups for group normalization')
parser.add_argument('--epochs', type=int, default=30,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--base_lr', type=float, default=0.007,
                    help='base learning rate')
parser.add_argument('--last_mult', type=float, default=1.0,
                    help='learning rate multiplier for last layers')
parser.add_argument('--scratch', action='store_true', default=False,
                    help='train from scratch')
parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='freeze batch normalization parameters')
parser.add_argument('--weight_std', action='store_true', default=False,
                    help='weight standardization')
parser.add_argument('--beta', action='store_true', default=False,
                    help='resnet101 beta')
parser.add_argument('--crop_size', type=int, default=513,
                    help='image crop size')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4,
                    help='number of model loading workers')
parser.add_argument('--model', type=str, default="deeplab",
                    help='model name')

args = parser.parse_args()



transform=transform_zoom_out



dataset = VOCSegmentation('C:/Users/hail/Desktop/ha/model/ADE/VOCdevkit',
                          train=args.train, crop_size=args.crop_size, process=None ,transform=transform)
valid_dataset = VOCSegmentation('C:/Users/hail/Desktop/ha/model/ADE/VOCdevkit',
                                train=not (args.train), crop_size=args.crop_size, process=None ,transform=transform)
import torchvision.transforms.functional as TF
print(dataset[0][0].size())
pil_image = TF.to_pil_image(dataset[3][0])

# PIL 이미지를 확인
pil_image.show()
