from datetime import datetime
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributed as dist
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from accelerate import Accelerator

class AdvancedTransformerConvNet(nn.Module):
    def __init__(self, num_classes=10, embed_dim=256, num_heads=8, num_layers=6, input_size=32):
        super(AdvancedTransformerConvNet, self).__init__()

        # Enhanced convolutional blocks with deeper structure and skip connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Residual block for improved gradient flow
        self.residual_block = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )

        # Squeeze-and-Excitation (SE) block for channel-wise attention
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512 // 16),
            nn.ReLU(),
            nn.Linear(512 // 16, 512),
            nn.Sigmoid()
        )

        # Compute flattened dimension dynamically
        reduced_size = input_size // 4  # Two MaxPool2d layers reduce size by factor of 4
        self.flatten_dim = reduced_size * reduced_size * 512
        self.embedding = nn.Linear(self.flatten_dim, embed_dim)

        # Transformer encoder layers with increased depth
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # Fully connected layers with advanced regularization
        self.fc1 = nn.Linear(embed_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Initial convolutional layers
        out = self.conv1(x)
        out = self.conv2(out)

        # Residual block with skip connection
        residual = out
        out = self.residual_block(out)
        out += residual  # Skip connection
        out = F.relu(out)

        # Squeeze-and-Excitation Block
        se_weight = self.se_block(out).unsqueeze(2).unsqueeze(3)  # Channel attention
        out = out * se_weight

        # Flatten and embed for Transformer
        batch_size = out.size(0)
        out = out.view(batch_size, -1)  # Flatten spatial dimensions
        out = self.embedding(out)  # Embed to transformer input dimension
        out = out.unsqueeze(1)  # Add sequence dimension for Transformer

        # Transformer layers
        out = self.transformer(out)

        # Pooling (mean over sequence length)
        out = out.mean(dim=1)

        # Fully connected layers
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)

        return out


def train(args):
    accelerator = Accelerator()
    # dist.init_process_group(backend='nccl')

    torch.manual_seed(0)
    # local_rank = int(os.environ['LOCAL_RANK'])
    # torch.cuda.set_device(local_rank)

    #verbose = dist.get_rank() == 0  # print only on global_rank==0
    verbose = accelerator.is_main_process  # print only in main process

    num_epochs = args.epochs

    #model = ConvNet().cuda()
    model = AdvancedTransformerConvNet(input_size=args.image_size).cuda()
    batch_size = 256

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # model = DistributedDataParallel(model, device_ids=[local_rank])

    train_dataset = CIFAR10(root='./data', train=True,
                          transform=transforms.ToTensor(), download=True)
    # train_sampler = DistributedSampler(train_dataset)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
    #                           shuffle=False, num_workers=0, pin_memory=True,
    #                           sampler=train_sampler)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    train_loader, model, optimizer = accelerator.prepare(train_loader, model,
                                                         optimizer)

    start = datetime.now()
    for epoch in range(num_epochs):
        tot_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # images = images.cuda(non_blocking=True)
            # labels = labels.cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            accelerator.backward(loss)
            optimizer.step()
            # loss.backward()
            optimizer.zero_grad()

            tot_loss += loss.item()

        if verbose:
            print('Epoch [{}/{}], average loss: {:.4f}'.format(
                epoch + 1,
                num_epochs,
                tot_loss / (i+1)))
    if verbose:
        print("Training completed in: " + str(datetime.now() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--image_size', default=32, type=int, metavar='N',
                        help='size of input images')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
