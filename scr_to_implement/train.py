import torch as t
import torch.optim
import os
from torch.utils.data import DataLoader
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model

import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data12 = 'data.csv'
#data12 = pd.read_csv(data)
#data = pd.read_csv(data12, sep=';')
csv_path = ''
for root, _, files in os.walk('.'):
    for name in files:
        if name == 'data.csv':
            csv_path = os.path.join(root, name)

data = pd.read_csv(csv_path, sep=';')
dataset_object = ChallengeDataset(data=data,mode='train')




# TODO
train_validation_ratio = 0.85
train_size = int(train_validation_ratio*len(dataset_object))
val_size = len(dataset_object) - train_size

train_set, val_set = torch.utils.data.random_split(dataset_object,[train_size,val_size])
train_loader = DataLoader(train_set,batch_size=32, shuffle=True,num_workers=0)
val_loader = DataLoader(val_set,batch_size=32,shuffle=True,num_workers=0)



# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO

# create an instance of our ResNet model
model = model.ResNet()
#print(model)
# TODO

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
criterion = t.nn.MultiLabelSoftMarginLoss()
# set up the optimizer (see t.optim)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# create an object of type Trainer and set its early stopping criterion
call_Trainer = Trainer(model=model, crit=criterion, optim=optimizer, train_dl=train_loader, val_test_dl=val_loader,cuda=False, early_stopping_patience=10)
# TODO

# go, go, go... call fit on trainer
res = call_Trainer.fit(100)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')

