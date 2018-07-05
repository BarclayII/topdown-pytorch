import torch as T
import skorch
from torchvision.datasets import MNIST
from topdown import *
from util import USE_CUDA, cuda

mnist = MNIST('.', download=True)

dfs = DFSGlimpseSingleObjectClassifier()
dfs.load_state_dict(T.load('bigmodel.pt'))

module = cuda(CNN(cnn='cnn', input_size=(15, 15), h_dims=128, n_classes=10, kernel_size=(3, 3), final_pool_size=(2, 2), filters=[16, 32, 64, 128, 256], pred=True))
#module.load_state_dict(T.load('cnn.pt'))
#module.load_state_dict(dfs.update_module.cnn.state_dict())

net = skorch.NeuralNetClassifier(
        module=module,
        #module=CNN,
        #module__cnn='cnn',
        #module__input_size=(15, 15),
        #module__h_dims=128,
        #module__n_classes=10,
        #module__kernel_size=(3, 3),
        #module__final_pool_size=(2, 2),
        #module__filters=[16, 32, 64, 128, 256],
        criterion=T.nn.CrossEntropyLoss,
        max_epochs=50,
        optimizer=T.optim.SGD,
        optimizer__param_groups=[
            ('cnn.*', {'lr': 0}),
            ('net_h.*', {'lr': 0}),
            ],
        lr=3e-5,
        batch_size=32,
        device='cuda' if USE_CUDA else 'cpu',
        callbacks=[
            skorch.callbacks.ProgressBar(),
            skorch.callbacks.Checkpoint('cnntest.pt'),
            ]
        )
train_data = cuda(mnist.train_data.float()[:, None].repeat(1, 3, 1, 1) / 255.)
print(module.forward(train_data[0:10]), mnist.train_labels[0:10])
net.fit(train_data, cuda(mnist.train_labels))
