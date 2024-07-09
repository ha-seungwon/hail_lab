import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import subprocess
from tqdm.auto import tqdm
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms, models
from pascal import VOCSegmentation
from utils import AverageMeter
import Classification_resnet
import PIGNet_GSPonly_classification
import torchvision

def get_cpu_temperature():
    sensors_output = subprocess.check_output("sensors").decode()
    for line in sensors_output.split("\n"):
        if "Tctl" in line:
            temperature_str = line.split()[1]
            temperature = float(temperature_str[:-3])  # remove "°C" and convert to float
            return temperature
    return None  # in case temperature is not found

def make_batch(samples, batch_size, feature_shape):
    inputs = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]
    if len(samples) < batch_size:
        num_padding = batch_size - len(samples)
        padding_tensor = torch.zeros(((num_padding,)+tuple(inputs[0].shape[:])))
        padded_inputs = torch.cat([torch.stack(inputs), padding_tensor], dim=0)
        padded_labels = torch.cat([torch.stack(labels), torch.zeros((num_padding,)+tuple(labels[0].shape[:]))], dim=0)
        return [padded_inputs, padded_labels]
    else:
        return [torch.stack(inputs), torch.stack(labels)]

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()
    for inputs, targets in tqdm(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
    return losses.avg

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    losses = AverageMeter()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100. * correct / total
    return losses.avg, accuracy

def main(args):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    if args.dataset == 'pascal':
        train_dataset = VOCSegmentation('C:/Users/hail/Desktop/ha/model/ADE/VOCdevkit', train=True, crop_size=args.crop_size, process=None)
        val_dataset = VOCSegmentation('C:/Users/hail/Desktop/ha/model/ADE/VOCdevkit', train=False, crop_size=args.crop_size, process=None)
    elif args.dataset == 'imagenet':
        data_dir = 'C:/Users/hail/Desktop/ha/data/Imagenet'
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        train_dataset = torchvision.datasets.ImageFolder(root=data_dir+'/train', transform=transform)
        val_dataset = torchvision.datasets.ImageFolder(root=data_dir+'/val', transform=transform)
    elif args.dataset in ['CIFAR-10', 'CIFAR-100']:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if args.dataset == 'CIFAR-10':
            train_dataset = torchvision.datasets.CIFAR10(root='path/to/cifar', train=True, download=True, transform=transform)
            val_dataset = torchvision.datasets.CIFAR10(root='path/to/cifar', train=False, download=True, transform=transform)
        else:
            train_dataset = torchvision.datasets.CIFAR100(root='path/to/cifar', train=True, download=True, transform=transform)
            val_dataset = torchvision.datasets.CIFAR100(root='path/to/cifar', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Load model
    if args.model == "Resnet":
        model = getattr(Classification_resnet, args.backbone)(
            pretrained=(not args.scratch),
            num_classes=len(train_dataset.classes),
            num_groups=args.groups,
            weight_std=args.weight_std,
            beta=args.beta
        )


    elif args.model == "PIGNet_GSPonly_classification":
        model = getattr(PIGNet_GSPonly_classification, args.backbone)(
            pretrained=(not args.scratch),
            num_classes=len(train_dataset.CLASSES),
            num_groups=args.groups,
            weight_std=args.weight_std,
            beta=args.beta,
            embedding_size=args.embedding_size,
            n_layer=args.n_layer,
            n_skip_l=args.n_skip_l)
    elif args.model == 'vit_b_16':
        model = models.vit_b_16(pretrained=True)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, len(train_dataset.classes))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)

    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        if val_acc > best_acc:
            print(f"Saving best model with accuracy: {val_acc:.2f}%")
            torch.save(model.state_dict(), 'best_model.pth')
            best_acc = val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False, help='training mode')
    parser.add_argument('--exp', type=str, default="bn_lr7e-3", help='name of experiment')
    parser.add_argument('--gpu', type=int, default=0, help='test time gpu device id')
    parser.add_argument('--backbone', type=str, default='resnet50', help='resnet50')
    parser.add_argument('--dataset', type=str, default='pascal', help='pascal or cityscapes')
    parser.add_argument('--groups', type=int, default=None, help='num of groups for group normalization')
    parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--base_lr', type=float, default=0.007, help='base learning rate')
    parser.add_argument('--last_mult', type=float, default=1.0, help='learning rate multiplier for last layers')
    parser.add_argument('--scratch', action='store_true', default=False, help='train from scratch')
    parser.add_argument('--freeze_bn', action='store_true', default=False, help='freeze batch normalization parameters')
    parser.add_argument('--weight_std', action='store_true', default=False, help='weight standardization')
    parser.add_argument('--beta', action='store_true', default=False, help='resnet101 beta')
    parser.add_argument('--crop_size', type=int, default=513, help='image crop size')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--workers', type=int, default=4, help='number of model loading workers')
    parser.add_argument('--model', type=str, default="deeplab", help='model name')
    args = parser.parse_args()

    args.model='Resnet'  #Resnet  PIGNet_GSPonly_classification  vit_b_16

    main(args)
