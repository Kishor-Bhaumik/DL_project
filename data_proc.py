import os
import torch
from torchvision import datasets
from torchvision import transforms

def prepare_imagenet(args):
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    kwargs = {} if args.no_cuda else {'num_workers': 1, 'pin_memory': True}

    # Pre-calculated mean & std on imagenet:
    # norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print('Preparing dataset ...')
    norm = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])

    train_trans = [
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        norm  # Include the normalization as part of the pipeline
    ]

    val_trans = [
        transforms.ToTensor(),
        norm  # Include the normalization as part of the pipeline
    ]


    train_data = datasets.ImageFolder(train_dir, transform=transforms.Compose(train_trans))
    val_data = datasets.ImageFolder(val_dir, transform=transforms.Compose(val_trans))

    print('Preparing data loaders ...')
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    return train_data_loader, val_data_loader
