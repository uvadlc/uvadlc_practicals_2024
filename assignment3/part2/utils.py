import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import copy
import time
import numpy as np
import matplotlib.pyplot as plt

from adversarial_attack import fgsm_loss, pgd_attack
from globals import STANDARD, FGSM, PGD


def load_cifar10(batch_size=4, valid_ratio=0.75, test_bs_1 = True, augmentations = False): 
    if augmentations:
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                   [0.247, 0.243, 0.261])])
    else:
        transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                               [0.247, 0.243, 0.261])])

    transform_validtest = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                   [0.247, 0.243, 0.261])])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    validtestset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_validtest)

    valid_len = int(len(validtestset) * valid_ratio)

    validset, testset = torch.utils.data.random_split(validtestset, [valid_len, len(validtestset) - valid_len])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1 if test_bs_1 else batch_size, shuffle=False, num_workers=2)

    classes = trainset.classes
    N_tr = len(trainset)
    N_tst = len(testset)
    N_vl = len(validset)

    attributes = {"class_names": classes, "N_train": N_tr, "N_test": N_tst, "N_valid": N_vl}

    return trainloader, validloader, testloader, attributes


def train(model, trainloader, validloader, num_epochs=25, defense_strategy = STANDARD, defense_args = {}):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    since = time.time()
    #define criterion
    criterion = nn.CrossEntropyLoss()
    #define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    #LR decay, 0.1, every 7 epoch
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    N_train = len(trainloader)*trainloader.batch_size
    N_valid = len(validloader)*validloader.batch_size
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = trainloader
                N=N_train
            else:
                model.eval()   # Set model to evaluate mode
                dataloader=validloader
                N=N_valid
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if defense_strategy == STANDARD or phase == 'val':
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    elif defense_strategy == FGSM and phase == 'train':
                        loss, preds = fgsm_loss(model, criterion, inputs, labels, 
                                                defense_args = defense_args,
                                                return_preds = True)
                    # backward + optimize only if in training phase
                    elif defense_strategy == PGD and phase == 'train':
                        # Get adverserial examples using PGD attack
                        # Add them to the original batch
                        # Make sure the model has the correct labels
                        raise NotImplementedError()
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(model(inputs), 1)

                    if phase == 'train':
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            if defense_strategy == PGD and phase == 'train':
                N*=2 #account for adverserial examples
            epoch_loss = running_loss / N
            epoch_acc = running_corrects.double() / N

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test(model, testloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test set: %d %%' % (
        100 * correct / total))
    return correct / total

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('image.png')

def show_image(dataloader, index):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    imshow(images[index])
    print('Label:', labels[index].item())