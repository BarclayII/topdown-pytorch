import torch as T
import skorch
from torchvision.datasets import MNIST
from datasets import MNISTMulti
from topdown import *
from util import USE_CUDA, cuda

mnist = MNIST('.', download=True)
#mnist = MNISTMulti('.', n_digits=1, backrand=0, image_rows=200, image_cols=200, download=True)

#dfs = DFSGlimpseSingleObjectClassifier()
#dfs.load_state_dict(T.load('bigmodel.pt'))

n_glimpses = 3

module = T.nn.Sequential(
        MultiscaleGlimpse(glimpse_type='gaussian', glimpse_size=(15, 15), n_glimpses=n_glimpses),
        CNN(cnn='cnn', input_size=(15, 15), h_dims=128, n_classes=10, kernel_size=(3, 3), final_pool_size=(1, 1), filters=[16, 32, 64, 128, 256], in_channels=3, pred=True, n_patches=n_glimpses, coalesce_mode='sample'),
        )
module = cuda(module)
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
        optimizer=T.optim.RMSprop,
        #optimizer__param_groups=[
        #    ('cnn.*', {'lr': 0}),
        #    ('net_h.*', {'lr': 0}),
        #    ],
        lr=3e-5,
        batch_size=32,
        device='cuda' if USE_CUDA else 'cpu',
        callbacks=[
            skorch.callbacks.ProgressBar(),
            skorch.callbacks.Checkpoint('cnntest.pt'),
            ]
        )
train_data = mnist.train_data.float()[:, None].repeat(1, 3, 1, 1) / 255.
#train_labels = mnist.train_labels[:, 0]
train_labels = mnist.train_labels
print(module.forward(cuda(train_data[0:10])), train_labels[0:10])
net.fit(train_data, train_labels)
