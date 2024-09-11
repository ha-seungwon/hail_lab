import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from PIL import Image
from torch.autograd import Variable
from tqdm.auto import tqdm
import pandas as pd
import os
from torchvision import transforms
import math
from model_src import Classification_resnet, PIGNet_GSPonly_classification, swin
import torch.nn.functional as F
from utils import AverageMeter
from torchvision.datasets import ImageFolder
from functools import partial
import torchvision
import subprocess
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import wandb

def make_batch_fn(samples, batch_size, feature_shape):
    return make_batch(samples, batch_size, feature_shape)

def visualize_compared_features(compared_features):
    plt.imshow(compared_features, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Compared Features')
    plt.show()

def compute_similarity(feature_map,labels):
    # feature_map의 shape을 가져옴
    batch_size, num_channels, height, width = feature_map.size()


    similarities = [0 for _ in range(4)]
    for channel_idx in range(num_channels):
        # 현재 채널(feature)의 특징 벡터
        channel_features = feature_map[:, channel_idx, :]

        # 채널의 중심(center) 지점
        center_idx = (int(height/2),int(width/2))
        center = channel_features[:, center_idx[0], center_idx[1]]  # 모든 데이터 포인트에 대해 중심 값을 가져옴
        distance_count=0
        # center에서 일정 거리만큼 떨어진 지점들의 코사인 유사도 계산
        distance = 2
        for z in range(4):  # distance가 1부터 4까지 변경 가능
            compared_indices = [(center_idx[0] + dx, center_idx[1] + dy) for dx in range(-distance, distance + 1) for dy
                                in range(-distance, distance + 1)]
            # 중복 제거
            compared_indices = list(set(compared_indices))

            # 이전 거리에 해당하는 인덱스 제외
            if distance > 1:
                previous_indices = [(center_idx[0] + dx, center_idx[1] + dy) for dx in range(-(distance - 1), distance)
                                    for dy
                                    in range(-(distance - 1), distance)]
                # 유효한 인덱스만 남기고 다른 것들 제거
                previous_indices = [(x, y) for (x, y) in previous_indices if 0 <= x < height and 0 <= y < width]
                compared_indices=[(x, y) for (x, y) in compared_indices if 0 <= x < height and 0 <= y < width]
                compared_indices = list(set(compared_indices) - set(previous_indices))



            # 현재 distance에 해당하는 지점들의 특징 벡터 가져오기
            compared = channel_features[:, [idx[0] for idx in compared_indices], [idx[1] for idx in compared_indices]]
            grap_compared = compared.cpu().squeeze()


            similarity = cosine_similarity(center, compared,dim=1)
            similarities[distance_count] += abs(similarity)
            distance_count+=1
            distance *= 2

    similarities_ = [similarity / num_channels for similarity in similarities]
    # 각 채널의 픽셀 유사도의 평균을 반환
    return similarities_

def get_cpu_temperature():
    sensors_output = subprocess.check_output("sensors").decode()
    for line in sensors_output.split("\n"):
        if "Tctl" in line:
            temperature_str = line.split()[1]
            temperature = float(temperature_str[:-3])  # remove "°C" and convert to float
            return temperature

    return None  # in case temperature is not found

def get_coords_by_distance(center_x, center_y, distance, feature_map_size_h, feature_map_size_w):
    coords = []
    for i in range(feature_map_size_h):
        for j in range(feature_map_size_w):
            dist = math.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            if abs(dist - distance) < 0.5:  # 거리 차이가 0.5 이내면 같은 거리로 간주
                coords.append((i, j))
    return coords


# 코사인 유사도 계산 함수
def calculate_cosine_similarity(coords, center_vector, tensor):
    cos_sims = []
    for (x, y) in coords:
        vector = tensor[0, :, x, y]  # 해당 좌표의 512차원 벡터
        cos_sim = F.cosine_similarity(center_vector, vector, dim=0)  # 코사인 유사도 계산
        cos_sims.append(cos_sim.item())
    return cos_sims

def zoom_center(image, zoom_factor):
    """
    Zooms into or out of the image around the center by the given zoom_factor.
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

        # Crop the image, then resize back to the original dimensions
        image = image.crop((left, top, right, bottom)).resize((width, height), Image.Resampling.LANCZOS)

    elif zoom_factor < 1:
        # Zoom out
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)

        # Resize the image to the new dimensions
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a new white image and paste the resized image in the center
        new_image = Image.new('RGB', (width, height), (255, 255, 255))
        new_image.paste(resized_image, ((width - new_width) // 2, (height - new_height) // 2))

        image = new_image

    return image
def repeat(image, pattern_repeat_count):
    """
    Repeat the inner region of the image in a grid pattern.
    pattern_repeat_count: Number of repetitions for each dimension (x, y)
    """
    image_size = image.size
    numpy_image = np.array(image)
    original_opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    # Use the entire image as the inner region to repeat
    inner_region = original_opencv_image
    inner_image = Image.fromarray(cv2.cvtColor(inner_region, cv2.COLOR_BGR2RGB))
    inner_image_resize = inner_image.resize((image_size[0], image_size[1]))

    # Create a new empty image of size (image_size[0] * repeat_count, image_size[1] * repeat_count)
    new_image_size = (image_size[0] * pattern_repeat_count, image_size[1] * pattern_repeat_count)
    new_image = Image.new('RGB', new_image_size)

    # Paste the resized inner image in a grid pattern
    for i in range(pattern_repeat_count):
        for j in range(pattern_repeat_count):
            new_image.paste(inner_image_resize, (i * image_size[0], j * image_size[1]))

    # Resize the final repeated image back to the original image size
    final_image = new_image.resize(image_size)

    return final_image
# Define a custom transform class for zoom
class ZoomTransform:
    def __init__(self, zoom_factor):
        self.zoom_factor = zoom_factor

    def __call__(self, image):
        return zoom_center(image, self.zoom_factor)


class RepeatTransform:
    def __init__(self, pattern_repeat_count):
        self.pattern_repeat_count = pattern_repeat_count

    def __call__(self, image):
        return repeat(image, self.pattern_repeat_count)





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
parser.add_argument('--process_type', type=str, default="zoom",
                    help='process_type')

args = parser.parse_args()
def model_size(model):
    total_size = 0
    for param in model.parameters():
        # °¢ ÆÄ¶ó¹ÌÅÍÀÇ ¿ø¼Ò °³¼ö °è»ê
        num_elements = torch.prod(torch.tensor(param.size())).item()
        # ¿ø¼Ò Å¸ÀÔ º°·Î ¹ÙÀÌÆ® Å©±â °è»ê (¿¹: float32 -> 4 bytes)
        num_bytes = num_elements * param.element_size()
        total_size += num_bytes
    return total_size
def make_batch(samples, batch_size, feature_shape):
    inputs = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]
    """
    print(inputs[0].shape)
    print(labels[0].shape)
    padding_tensor = torch.zeros(((2,) + tuple(inputs[0].shape[:])))
    print(padding_tensor.shape,torch.stack(inputs).shape)
    padded_inputs = torch.cat([torch.stack(inputs), padding_tensor], dim=0)
    print(padded_inputs.shape)
    padded_labels = torch.cat([torch.stack(labels), torch.zeros((2,)+tuple(labels[0].shape[:]))], dim=0)
    print(padded_labels.shape)"""
    if len(samples) < batch_size:

        num_padding = batch_size - len(samples)
        padding_tensor = torch.zeros(((num_padding,)+tuple(inputs[0].shape[:])))
        padded_inputs = torch.cat([torch.stack(inputs), padding_tensor], dim=0)

        padded_labels = torch.cat([torch.stack(labels), torch.zeros((num_padding,)+tuple(labels[0].shape[:]))], dim=0)
        return [padded_inputs, padded_labels]
    else:

        return [torch.stack(inputs), torch.stack(labels)]


def main():
    # make fake args
    args = argparse.Namespace()
    args.dataset = "imagenet" #CIFAR-10 CIFAR-100  imagenet
    args.model = "PIGNet_GSPonly_classification" #Resnet  PIGNet_GSPonly_classification  vit_b_16  swin
    args.backbone = "resnet101"
    args.workers = 4
    args.epochs = 50
    args.batch_size = 8
    args.train = False
    args.crop_size = 513 #513
    args.base_lr = 0.007
    args.last_mult = 1.0 
    args.groups = None
    args.scratch = False
    args.freeze_bn = False
    args.weight_std = False
    args.beta = False
    args.resume = None
    args.exp = "bn_lr7e-3"
    args.gpu = 0
    args.embedding_size = 512
    args.n_layer = 6
    args.n_skip_l = 2 #2
    args.process_type = "zoom"  #zoom overlap repeat None
    #pattern_repeat_count = 2
    zoom_factor = 1.5

    # if is cuda available device
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    print("ags",args.model,args.dataset)
    print("cuda available", args.device)

    if args.train:
        wandb.init(project='pignet_classification', name=args.model+'_'+args.backbone+ '_embed' + str(args.embedding_size) +'_nlayer' + str(args.n_layer) + '_'+args.exp,
                    config=args.__dict__)

    loss_data = pd.DataFrame(columns=["train_loss"])
    loss_list = []

    # assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    model_fname = 'model/classification_{0}_{1}_{2}_v3.pth'.format(
        args.model,args.backbone, args.dataset)

    if args.dataset == 'imagenet':
        # 데이터셋 경로 및 변환 정의
        image_size=224
        data_dir = 'C:/Users/hail/Desktop/ha/data/Imagenet'
        # Set the zoom factor (e.g., 1.2 to zoom in, 0.8 to zoom out)

        if args.train:
            # Define transformations
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),  # Resize to fixed size
                transforms.ToTensor(),  # Convert image to tensor
            ])
        else:
            if args.process_type==None:
                transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),  # Resize to fixed size
                    transforms.ToTensor(),  # Convert image to tensor
                ])
            else:
                if args.process_type == 'zoom':
                    # Define transformations
                    transform = transforms.Compose([
                        transforms.Resize((image_size, image_size)),  # Resize to fixed size
                        ZoomTransform(zoom_factor),  # Apply the zoom transformation
                        transforms.ToTensor(),  # Convert image to tensor
                    ])
                elif args.process_type =='repeat':



                    # Define transformations
                    transform = transforms.Compose([
                        transforms.Resize((image_size, image_size)),  # Resize to fixed size
                        RepeatTransform(pattern_repeat_count),  # Apply the repeat transformation
                        transforms.ToTensor(),  # Convert image to tensor
                    ])

        # Load datasets with ImageFolder and apply transformations
        dataset = ImageFolder(root=f'{data_dir}/train', transform=transform)
        valid_dataset = ImageFolder(root=f'{data_dir}/val', transform=transform)


        # dataset = torchvision.datasets.ImageFolder(root=data_dir+'/train', transform=transform)
        #
        # valid_dataset = torchvision.datasets.ImageFolder(root=data_dir+'/val', transform=transform)
        idx2label = []
        cls2label = {}

        import json
        json_file=data_dir+'/imagenet_class_index.json'
        with open(json_file, "r") as read_file:
            class_idx = json.load(read_file)
            idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
        dataset.CLASSES = idx2label

    elif args.dataset == 'CIFAR-100':
        image_size = 32

        if args.train:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지를 정규화합니다.
            ])
        else:
            if args.process_type==None:
                print("original data")
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지를 정규화합니다.
                ])
            else:
                if args.process_type == 'zoom':
                    transform = transforms.Compose([
                        ZoomTransform(zoom_factor),  # Apply the zoom transformation
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 이미지를 정규화합니다.
                    ])
                elif args.process_type == 'repeat':



                    # Define transformations
                    transform = transforms.Compose([
                        transforms.Resize((image_size, image_size)),  # Resize to fixed size
                        RepeatTransform(pattern_repeat_count),  # Apply the repeat transformation
                        transforms.ToTensor(),  # Convert image to tensor
                    ])



        # CIFAR-100 데이터셋 로드
        dataset = torchvision.datasets.CIFAR100(root='C:/Users/hail/Desktop/ha/model', train=True, download=True, transform=transform)
        valid_dataset = torchvision.datasets.CIFAR100(root='C:/Users/hail/Desktop/ha/model', train=False, download=True, transform=transform)

        dataset.CLASSES=sorted(['beaver', 'dolphin', 'otter', 'seal', 'whale',  # aquatic mammals
                           'aquarium' 'fish', 'flatfish', 'ray', 'shark', 'trout',  # fish
                           'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', # flowers
                           'bottles', 'bowls', 'cans', 'cups', 'plates', # food containers
                           'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', # fruit and vegetables
                           'clock', 'computer' 'keyboard', 'lamp', 'telephone', 'television', # household electrical devices
                           'bed', 'chair', 'couch', 'table', 'wardrobe', # household furniture
                           'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', # insects
                           'bear', 'leopard', 'lion', 'tiger', 'wolf', # large carnivores
                           'bridge', 'castle', 'house', 'road', 'skyscraper', # large man-made outdoor things
                           'cloud', 'forest', 'mountain', 'plain', 'sea', # large natural outdoor scenes
                           'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', # large omnivores and herbivores
                           'fox', 'porcupine', 'possum', 'raccoon', 'skunk', # medium-sized mammals
                           'crab', 'lobster', 'snail', 'spider', 'worm', # non-insect invertebrates
                           'baby', 'boy', 'girl', 'man', 'woman', # people
                           'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', # reptiles
                           'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', # small mammals
                           'maple', 'oak', 'palm', 'pine', 'willow', # trees
                           'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', # vehicles 1
                           'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor' # vehicles 2
                          ])
    elif args.dataset == 'CIFAR-10':

        transform = transforms.Compose([
            transforms.ToTensor()])

        # CIFAR-100 데이터셋 로드
        dataset = torchvision.datasets.CIFAR10(root='C:/Users/hail/Desktop/ha/model', train=True, download=True,transform=transform)#, transform=transform,target_transform=transform_target)
        valid_dataset = torchvision.datasets.CIFAR10(root='C:/Users/hail/Desktop/ha/model', train=False, download=True,transform=transform)#, transform=transform,target_transform=transform_target)
        dataset.CLASSES =['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.backbone in ["resnet50","resnet101","resnet152"]:
        if args.model == "PIGNet_GSPonly_classification":
            model = getattr(PIGNet_GSPonly_classification, args.backbone)(
                pretrained=(not args.scratch),
                num_classes=len(dataset.CLASSES),
                num_groups=args.groups,
                weight_std=args.weight_std,
                beta=args.beta,
                embedding_size = args.embedding_size,
                n_layer = args.n_layer,
                n_skip_l = args.n_skip_l)
            print("model PIGNet_GSPonly_classification")
        elif args.model == "Resnet":
            print("Classification_resnet model load")
            model = getattr(Classification_resnet, args.backbone)(
                pretrained=(not args.scratch),
                num_classes=len(dataset.CLASSES),
                num_groups=args.groups,
                weight_std=args.weight_std,
                beta=args.beta,
                embedding_size=args.embedding_size,
                n_layer=args.n_layer,
                n_skip_l=args.n_skip_l
                )
        elif args.model == 'vit_b_16':
            print("main")
        elif args.model == 'swin':
            model = swin.SwinTransformer(img_size=image_size, num_classes=len(dataset.CLASSES), window_size=4) #window_size =4 cifar   img_size=

    else:

        raise ValueError('Unknown backbone: {}'.format(args.backbone))
    size_in_bytes = model_size(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print number of parameters
    print(f"Number of parameters: {num_params / (1000.0 ** 2): .3f} M")

    # num_params_gnn = sum(p.numel() for p in model.pyramid_gnn.parameters() if p.requires_grad)
    # print(f"Number of GNN parameters: {num_params_gnn / (1000.0 ** 2): .3f} M")


    print(f"Entire model size: {size_in_bytes / (1024.0 ** 3): .3f} GB")

    print("train",args.train)

    # Initialize a DataFrame to store the training log
    #train_log = pd.DataFrame(columns=["epoch", "train_loss", "train_accuracy"])
    if args.train:
        print("Training !!! ")
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        model = nn.DataParallel(model).to(args.device)
        model.train()
        if args.freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    print("12345")
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        if args.model!='swin':
            backbone_params = (
                        list(model.module.conv1.parameters()) +
                        list(model.module.bn1.parameters()) +
                        list(model.module.layer1.parameters()) +
                        list(model.module.layer2.parameters()) +
                        list(model.module.layer3.parameters()) +
                        list(model.module.layer4.parameters()))

            if args.model == "PIGNet_GSPonly_classification":
                last_params = list(model.module.pyramid_gnn.parameters())
                optimizer = optim.SGD([
                    {'params': filter(lambda p: p.requires_grad, backbone_params)},
                    {'params': filter(lambda p: p.requires_grad, last_params)}

                ],lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
            elif args.model == "ASPP":
                last_params = list(model.module.aspp.parameters())
                optimizer = optim.SGD([
                    {'params': filter(lambda p: p.requires_grad, backbone_params)},
                    {'params': filter(lambda p: p.requires_grad, last_params)}

                ],lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

            else:
                last_params = list(model.module.linear.parameters())
                optimizer = optim.SGD([
                    {'params': filter(lambda p: p.requires_grad, backbone_params)},
                    {'params': filter(lambda p: p.requires_grad, last_params)}

                ],
                    lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

        else:# swin
            last_params = list(model.module.parameters())
            optimizer = optim.SGD([
                {'params': filter(lambda p: p.requires_grad, last_params)}

            ], lr=0.001, momentum=0.9, weight_decay=0.0001)

        feature_shape = (2048,33,33)


        collate_fn = partial(make_batch_fn, batch_size=args.batch_size, feature_shape=feature_shape)

        dataset_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.train
        )
        valid_dataset = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False
        )

        max_iter = args.epochs * len(dataset_loader)
        losses = AverageMeter()
        start_epoch = 0

        if args.resume:
            if os.path.isfile(args.resume):
                print('=> loading checkpoint {0}'.format(args.resume))
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('=> loaded checkpoint {0} (epoch {1})'.format(
                    args.resume, checkpoint['epoch']))
            else:
                print('=> no checkpoint found at {0}'.format(args.resume))


        best_loss = 1e+10
        patience = 0

        train_step = 0

        for epoch in range(start_epoch, args.epochs):
            print("EPOCHS : ", epoch+1," / ",args.epochs)

            loss_sum = 0
            acc_sum = 0
            cnt = 0

            for inputs, target in tqdm(iter(dataset_loader)):
                log = {}
                cur_iter = epoch * len(dataset_loader) + cnt
                lr = args.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
                optimizer.param_groups[0]['lr'] = lr
                # if args.model == "deeplab":
                inputs = Variable(inputs.to(args.device))
                target = Variable(target.to(args.device)).long()
                if args.model=='swin':
                    outputs = model(inputs)

                else:
                    outputs,_ = model(inputs)



                outputs_flat = outputs.view(-1, outputs.size(-1))  # 배치 크기 유지하고 마지막 차원을 평탄화

                # 크로스 엔트로피 손실 계산
                loss = F.cross_entropy(outputs_flat, target)
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    pdb.set_trace()
                losses.update(loss.item(), args.batch_size)
                loss_sum += loss.item()
                loss.backward()
                #print('batch_loss: {:.5f}'.format(loss))

                # calculate accuracy
                _, predicted = outputs.max(1)
                total = target.size(0)
                correct = predicted.eq(target).sum().item()
                accuracy = 100 * correct / total
                acc_sum += accuracy
                log['train/batch/accuracy'] = accuracy
                log['train/batch/loss'] = loss.to('cpu')
                #log['train/temp'] = get_cpu_temperature()
                wandb.log(log)


                cnt+=1
                train_step += 1

                optimizer.step()
                optimizer.zero_grad()

                # if train_step % 30 == 0:
                #     time.sleep(15)

            log = {}
            loss_avg = loss_sum/len(dataset_loader)
            acc_avg = acc_sum/len(dataset_loader)
            log['train/epoch/loss'] = loss_avg
            log['train/epoch/accuracy'] = acc_avg

            wandb.log(log)



            loss_list.append(loss_avg)
            print('epoch: {0}\t'
                  'lr: {1:.6f}\t'
                  'loss: {2:.4f}'.format(
                epoch + 1, lr,loss_avg))



            if best_loss > loss_avg:
                best_loss = loss_avg
                patience = 0

                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, model_fname)
                print("model save complete!!!")
            else:
                patience +=1
                print("patience : ",patience)

                if patience > 50:
                    break
            # time.sleep(150)
            print("Evaluating !!! ")


            with torch.no_grad():
                model.eval()
                losses_test = 0.0
                correct = 0
                total = 0

                for i, (inputs, labels) in enumerate(tqdm(valid_dataset)):
                    inputs = inputs.to(args.device)
                    labels = torch.tensor(labels).to(args.device)
                    if args.model == 'swin':
                        outputs = model(inputs)

                    else:
                        outputs, _ = model(inputs)

                    loss = criterion(outputs, labels)

                    losses_test += loss.item()

                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                accuracy = 100 * correct / total
                print('Accuracy: {:.2f}%'.format(accuracy))

                # train_log = train_log.append({"epoch": epoch, "train_loss":  losses_test / len(valid_dataset), "train_accuracy": accuracy},
                #                              ignore_index=True)


                log['test/epoch/loss'] = losses_test / len(valid_dataset)
                log['test/epoch/accuracy'] = accuracy

            wandb.log(log)
            # time.sleep(60)
            model.train()

        # train_log.to_csv("training_log.csv", index=False)
    else:
        print("Evaluating !! ")

        torch.cuda.set_device(args.gpu)
        model = model.to(args.device)
        model.eval()
        checkpoint = torch.load(model_fname)

        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
        model.load_state_dict(state_dict)

        valid_dataset = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False
        )

        with torch.no_grad():
            model.eval()
            losses_test = 0.0
            correct = 0
            total = 0


            distances = [1, 2, 3, 4]
            distances_sum= [ 0 for _ in range(len(distances))]

            for i, (inputs, labels) in enumerate(tqdm(valid_dataset)):
                #
                # import matplotlib.pyplot as plt
                # input_image = inputs.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                # plt.figure(figsize=(6, 6))
                # plt.imshow(input_image)  # 원본 데이터 그대로 출력 (float32 지원)
                # plt.title('Input Image (Original)')
                # plt.axis('off')
                # plt.show()


                inputs = inputs.to(args.device)
                labels = torch.tensor(labels).to(args.device)
                outputs, layer_outputs = model(inputs)



                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # 텐서 크기에서 중심 좌표를 자동으로 설정 (H/2, W/2)
            # H, W = layer_outputs.shape[2], layer_outputs.shape[3]
            # center_x, center_y = H // 2, W // 2
            #
            # 중심점 feature 벡터 (512차원)
            # center_vector = layer_outputs[0, :, center_x, center_y]
            #
            # 각 거리에 대해 코사인 유사도 계산
            # for distance in distances:
            #     coords = get_coords_by_distance(center_x, center_y, distance, H, W)  # 거리별 좌표 구하기
            #     cos_sims = calculate_cosine_similarity(coords, center_vector, layer_outputs)  # 유사도 계산
            #     mean_cos_sim = sum(cos_sims) / len(cos_sims)  # 평균 코사인 유사도 계산
            #     index = distances.index(distance)
            #     distances_sum[index] += mean_cos_sim


            #for idx, model in enumerate(pixel_similarity_value):
            #    for idx_, data_ in enumerate(model):
            #        pixel_similarity_value[idx][idx_]=data_/count
            accuracy = 100 * correct / total
            print('Accuracy: {:.2f}%'.format(accuracy))
            for i, distance in enumerate(distances):
                print(f"Sum of Cosine similarity for distance {distance}: {distances_sum[i]/len(valid_dataset)}")



if __name__ == "__main__":
    main()
