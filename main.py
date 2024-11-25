#!/usr/bin/env python
# coding: utf-8


import os
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torchvision.models as models
from collections import defaultdict
from data_proc import prepare_imagenet


# torch.backends.cudnn.enabled = False


parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Training')

parser.add_argument('--dataset', default='tiny-224', choices=['mnist', 'tiny-224'],  help='name of dataset to train on (default: tiny-imagenet-200)')
parser.add_argument('--data-dir', default=os.getcwd(), type=str,   help='path to dataset (default: current directory)')
parser.add_argument('--batch-size', default=128, type=int,   help='mini-batch size for training (default: 1000)')
parser.add_argument('--test-batch-size', default=128, type=int,   help='mini-batch size for testing (default: 1000)')
parser.add_argument('--epochs', default=25, type=int,   help='number of total epochs to run (default: 25)')
parser.add_argument('--seed', default=1, type=int,      help='seed for initializing training (default: 1)')
parser.add_argument('--no-cuda', action='store_true',   help='run without cuda (default: False)')
parser.add_argument('--log-interval', default=100, type=int,   help='batches to wait before logging detailed status (default: 100)')
parser.add_argument('--model', default='AlexNet', choices=['SVM', 'AlexNet'],   help='model to train (default: AlexNet)')
parser.add_argument('--pretrained', action='store_true',    help='use pretrained AlexNet model (default: False)')
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'],   help='optimizer (default: adam)')
parser.add_argument('--momentum', default=0.5, type=float,    help='momentum (default: 0.5)')
parser.add_argument('--lr', default=0.01, type=float,    help='learning rate (default: 0.01)')
parser.add_argument('--classes', default=200, type=int,  help='number of output classes of SVM (default: 200)')
parser.add_argument('--reg', action='store_true',   help='add L2 regularization for hinge loss (default: False)')
parser.add_argument('--margin', default=20, type=int,   help='margin for computing hinge loss (default: 20)')
parser.add_argument('--topk', default=1, type=int,  help='top-k accuracy (default: 1)')
parser.add_argument('--results-dir', default=os.path.join(os.getcwd(), 'results'), type=str,  help='path to plots (default: cwd/results)')
parser.add_argument('--save', action='store_true',  help='save model (default: False)')
parser.add_argument('--models-dir', default=os.path.join(os.getcwd(), 'models'), type=str,   help='path to save model (default: cwd/models)')
parser.add_argument('--load', action='store_true',  help='load model (default: False)')
parser.add_argument('--model-path', default=os.path.join(os.getcwd(), 'models', 'default.pt'), type=str,  help='path to load model (default: cwd/models/default.pt)')
parser.add_argument('--err', action='store_true',  help='plot error analysis graphs (default: False)')
parser.add_argument('--use_wandb', action='store_true', default= False,   help='log with wandb ')

args = parser.parse_args()
if not args.no_cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = prepare_imagenet(args)


model = models.resnet18(weights=False)
model.fc = nn.Linear(512, 200)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
experiment = 'resent18'
if args.use_wandb:
    import wandb
    wandb.init(project="finalProj_DL", name=experiment)
    wandb.require("core")


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(init_weights)

# Test the model
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()
    
    acc = correct / total
    return acc


def train(model, train_loader, criterion, optimizer, num_epochs=args.epochs):
    model.train()
    
    train_losses = []
    train_accs = []
    test_accs = []

        
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        test_acc = test(model, test_loader)
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        test_accs.append(test_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Test Acc: {test_acc:.4f}")

        if args.use_wandb:
            wandb.log({"Train Accuracy :": epoch_acc, "Test Accuracy :": test_acc, "Train loss": epoch_loss})
        
    return train_losses, train_accs, test_accs


train_losses, train_accs, test_accs = train(model.to(device), train_loader, criterion, optimizer)
