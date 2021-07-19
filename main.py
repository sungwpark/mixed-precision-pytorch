import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import ResNet, BasicBlock
from train import train, validate


def main():
    # Data loading code
    train_dataset = datasets.MNIST(root='./MNIST_data',
        train=True, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(train_dataset, batch_size=64)

    val_dataset = datasets.MNIST(root='./MNIST_data',
        train=False, transform=transforms.ToTensor(), download=True)

    val_loader = DataLoader(val_dataset, batch_size=64)

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
        train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)
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