import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import ResNet, BasicBlock
from train import train, validate
from utils import save_checkpoint


def main():
    ## Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                        (4,4,4,4),mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])
        
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 2, 'pin_memory': True}

    train_loader = DataLoader(
        datasets.__dict__['CIFAR10']('../data', train=True, download=True, transform=transform_train),
        batch_size=128, shuffle=True, **kwargs)
    val_loader = DataLoader(
        datasets.__dict__['CIFAR10']('../data', train=False, transform=transform_test),
        batch_size=128, shuffle=False, **kwargs)

    # create model
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=200)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # cosine learning rate
    epochs = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*epochs)
    scaler = torch.cuda.amp.GradScaler()

    cudnn.benchmark = True

    #Training
    best_prec1 = 0
    writer = SummaryWriter("ResNet34")

    for epoch in range(epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, writer)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, writer)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)

if __name__ == '__main__':
    main()