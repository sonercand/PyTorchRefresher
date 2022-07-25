import enum
import torch
import numpy as np
from torch import nn

cats = torch.load("./PyTorchRefresher/data/train.csv/cats.pt")
conts = torch.load("./PyTorchRefresher/data/train.csv/conts.pt")
label = torch.load("./PyTorchRefresher/data/train.csv/label.pt")
# cats data - 24 hr, [1,0] am , 7 days
# set embedding sizes
embS = [(24, 12), (2, 1), (7, 4)]  # take halves as embedding dimensions -> output
embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embS])
print(embeds)
sample = cats[:2]
print(sample)
# sample forward pass
embs = []
for i, emb in enumerate(embeds):
    embs.append(emb.forward(sample[:, i]))
print(embs)
layer = torch.cat(embs, 1)
print(layer)


class Model(nn.Module):
    def __init__(self, emb_sizes, n_cont, output_size, layers, p=0.5):
        """
        args:
            emb_sizes: list or tuples of embedding sizes. Eg [(24,12),(2,1),(7,4)] where 24,2,7 are the number of cats in cat variables whereas 12,1,4 are the embedding output sizes
            n_cont: number of continues variables
            output_size: size output layer
            layers: list of layers containing node sizes of each layer
        """
        super().__init__()
        self.embeding = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_sizes])
        self.embeding_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)  # normalise continues variables
        layerList = []
        n_embs = sum(
            [nf for ni, nf in emb_sizes]
        )  # total number of nodes in embeding output
        number_of_input_nodes = n_embs + n_cont
        for i in layers:
            layerList.append(nn.Linear(number_of_input_nodes, i))
            layerList.append(nn.ReLU(inplace=True))
            layerList.append(nn.BatchNorm1d(i))
            layerList.append(nn.Dropout(p))
            number_of_input_nodes = i
        layerList.append(nn.Linear(layers[-1], output_size))
        self.layers = nn.Sequential(*layerList)

    def forward(self, x_cat, x_cont):
        # cats first
        embedding = []
        for i, emb in enumerate(self.embeding):
            embedding.append(emb(x_cat[:, i]))
        x = torch.cat(embedding, axis=1)
        x = self.embeding_drop(x)
        # cont
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], axis=1)
        x = self.layers(x)
        return x


torch.manual_seed(12)
embS = [(24, 12), (2, 1), (7, 4)]
print(conts.shape[1])
model = Model(
    emb_sizes=embS,
    n_cont=conts.shape[1],
    output_size=1,
    layers=[32, 16, 8, 4, 2],
    p=0.35,
)
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(conts.shape)
# data set split
test_size = int(conts.shape[0] * 0.2)
print(test_size)
train_cat = cats[: cats.shape[0] - test_size]
test_cat = cats[cats.shape[0] - test_size :]
train_cont = conts[: cats.shape[0] - test_size]
test_cont = conts[cats.shape[0] - test_size :]
train_label = label[: cats.shape[0] - test_size]
test_label = label[cats.shape[0] - test_size :]
print(train_cat.shape, train_cont.shape, train_label.shape)
print(test_cat.shape, test_cont.shape, test_label.shape)
# train the model
epochs = 200
losses = []
for i in range(epochs):
    y_pred = model.forward(train_cat, train_cont)
    loss = criterion(y_pred, train_label)
    losses.append(loss)
    if i % 10 == 0:
        print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
import matplotlib.pyplot as plt

with torch.no_grad():
    y_test_pred = model(test_cat, test_cont)
    loss_test = criterion(y_test_pred, test_label)
    print(loss_test)
plt.scatter(y_test_pred, test_label)
plt.show()
print(y_test_pred[0:10], test_label[0:10])
