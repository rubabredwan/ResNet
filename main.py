import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from resnet import ResNet20, ResNet32, ResNet44, ResNet56

epochs = 200
batch_size = 128
lr = 0.1
momentum = 0.9
weight_decay = 0.0001
path = 'pretrained/'

if not os.path.exists(path):
    os.makedirs(path)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=batch_size, shuffle=True,
    num_workers=4)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
,   batch_size=batch_size, shuffle=False,
    num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


for net in (ResNet20, ResNet32, ResNet44, ResNet56):
    
    model = net()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                        milestones=[100, 150], 
                                                        last_epoch=-1)
    history = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}

    def load_checkpoint()
        checkpoint = torch.load(path+net.__name__+'.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        history = checkpoint['history']
        start_epoch = checkpoint['epoch']

    def save_checkpoint(epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'history':history
            }, path + net.__name__ + '.pth')


    if os.path.exists(path+net.__name__+'.pth'):
        load_checkpoint()
    
    def train():
        model.train()

        running_loss = 0
        correct = 0
        total = 0

        start = time.time()
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)
            _, predicted = torch.max(output.data, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return correct/total * 100, running_loss / 50000 * 128 

    def validate():
        model.eval()

        running_loss = 0
        correct = 0
        total = 0
        start = time.time()
        
        for i, (images, labels) in enumerate(val_loader):

            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)
            _, predicted = torch.max(output.data, 1)

            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return correct / total * 100, running_loss / 10000 * 128


    for epoch in range(start_epoch, epochs):
        start = time.time()
        acc, loss = train()
        val_acc, val_loss = validate()

        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('%.2f - loss: %.4f - acc: %.2f - val_loss: %.4f - val_acc: %.2f' % (
            time.time() - start,
            loss, acc, val_loss, val_acc))

        history['acc'] += [acc]
        history['loss'] += [loss]
        history['val_acc'] += [val_acc]
        history['val_loss'] += [val_loss]

        lr_scheduler.step()

        if epoch % 20 == 0:
            save_checkpoint(epoch+1)

    save_checkpoint(epochs)