
import time
import torch


import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch import optim
from torch.utils.data import DataLoader

from loss import ContrastiveLoss

from config_model import ConfigModel
from dataset import SiameseNetworkDataset
from siamese_net_3cnn import SiameseNetwork

from tools import show_plot


def train():
    print(torch.cuda.is_available())
    begin = time.time()

    folder_dataset = dset.ImageFolder(root=ConfigModel.training_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                          transforms.ToTensor()]), should_invert=False)

    train_data_loader = DataLoader(siamese_dataset, shuffle=True, num_workers=6, batch_size=ConfigModel.train_batch_size)

    net = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    counter = []
    loss_history = []
    iteration_number = 0
    for epoch in range(0, ConfigModel.train_number_epochs):
        for i, data in enumerate(train_data_loader):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

    show_plot(counter, loss_history)
    end = time.time()
    consume = end-begin
    print('time consume：{}'.format(consume))
    torch.save(net.state_dict(), ConfigModel.save_model)


if __name__ == '__main__':
    train()
