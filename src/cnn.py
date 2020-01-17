import torch as T
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

def cnn():
    trainData = datasets.CIFAR10(root='data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
    testData = datasets.CIFAR10(root='data',
                               train=False,
                               transform=transforms.ToTensor(),
                               download=True)

    ## working with CIFAR dataset

    # print(trainData.train_data.size())
    # print(trainData.train_labels.size())

    batchsize = 100
    nIter = 3000
    numEpochs = nIter / (len(trainData) / batchsize)
    numEpochs = int(numEpochs)

    trainDataLoader = T.utils.data.DataLoader(dataset = trainData,
                                              batch_size = batchsize,
                                              shuffle=True)
    trainDataLoader = T.utils.data.DataLoader(dataset = testData,
                                              batch_size = batchsize,
                                              shuffle=False)
    iters = iter(trainDataLoader)
    images, labes = iters.next()
    print(images.shape)

    ## visualizing data
    image_data = images[0] 
    print(image_data.shape)
    np_image = image_data.numpy()
    np_image = np.transpose(np_image, (1,2, 0))
    print(np_image.shape)
    plt.imshow(np_image)
    plt.show()
    




    pass

## model
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()


def run():
    '''
    entry point for the file
    :return: none
    '''
    cnn()
    pass
