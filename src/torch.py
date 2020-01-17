import torch as T
import numpy as np
import torch.nn as nn


def run():
    '''
    this is the entry function for our program
    '''
    # testTorchGpu()
    # createTensors()
    # autoDiff()
    lr()
    pass


def testTorchGpu():
    print(T.cuda.is_available())


def prtMe(inp1, inp2):
    msg = inp1 + " {}"
    print(msg.format(inp2))


def createTensors():
    prtMe("hello", 2)
    t1 = T.tensor([1, 2, 3])
    prtMe("tensor through list", t1)
    n1 = np.random.random((2, 2))
    prtMe("np array", n1)
    t2 = T.from_numpy(n1)
    prtMe("tensor from numpy ", t2)
    prtMe("tensor datatype", t2.dtype)


def autoDiff():
    x = T.tensor([5], dtype=T.float32, requires_grad=True)
    y = T.tensor([6], dtype=T.float32, requires_grad=True)
    print(x)
    print(y)
    z = ((x ** 2) * y) + (x * y)
    print(x)

    ## to find the derivate with respect to all varibles
    # autograd to be applied on scalar
    total = T.sum(z)
    print(total)
    total.backward()
    print(x.grad)
    print(y.grad)


def fullConnectedLayer():
    import torch.nn as nn
    model = nn.Sequential(nn.Linear(10, 5),
                          nn.ReLU(),
                          nn.Linear(5, 1),
                          nn.Sigmoid())
    mse = nn.MSELoss()
    opt = T.optim.SGD(model.parameters(), lr=0.01)


## building linear regression model
class LinearRegression(nn.Module):
    def __init__(self, inputDim, outputDim):
        super().__init__()
        self.linear = nn.Linear(inputDim, outputDim)

    def forward(self, input):
        out = self.linear(input)
        return out

def lr():
    device = 'cuda'
    inputDim = 1
    outputDim = 1
    x = [i for i in range(10)]
    inp = np.array(x, np.float32)
    inp = inp.reshape(-1, 1)
    print(inp.shape)
    inputs = T.from_numpy(inp).requires_grad_()
    y = np.array(inp ** 2, dtype=np.float32)
    y = y.reshape(-1, 1)
    labels = T.tensor(y)
    print(inputs)
    print(inputs.shape)
    model = LinearRegression(inputDim, outputDim)
    model = model.to(device)
    criterian = nn.MSELoss()
    lr = 0.01
    optimizer = T.optim.SGD(model.parameters(), lr)
    inputs = inputs.to(device)
    labels = labels.to(device)
    epochs = 100000
    for i in range(epochs):
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterian(output, labels)

        loss.backward()

        optimizer.step()

        print('epoch {}, loss {}'.format(i, loss.item()))

