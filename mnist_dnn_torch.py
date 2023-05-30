import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data as td
torch.manual_seed(0)

x_train = 'train-images.idx3-ubyte'
x_train = idx2numpy.convert_from_file(x_train)/255.0
print('X_train shape: ',x_train.shape)

y_train = 'train-labels.idx1-ubyte'
y_train = idx2numpy.convert_from_file(y_train)
print('y_train shape: ',y_train.shape)

x_test = 't10k-images.idx3-ubyte'
x_test = idx2numpy.convert_from_file(x_test)/255.0
print('X_test shape: ',x_test.shape)

y_test = 't10k-labels.idx1-ubyte'
y_test = idx2numpy.convert_from_file(y_test)
print('y_test shape: ',y_test.shape)

X_train = x_train.reshape(x_train.shape[0],-1)
X_test = x_test.reshape(x_test.shape[0],-1)

########################################################################################################################
#CONVERT NUMPY ARRAY TO TORCH TENSORS
train_x = torch.tensor(X_train).float()
train_y = torch.tensor(y_train).long()
train_ds = td.TensorDataset(train_x, train_y)
train_loader = td.DataLoader(train_ds, batch_size=64, shuffle=True)

test_x = torch.tensor(X_test).float()
test_y = torch.tensor(y_test).long()
test_ds = td.TensorDataset(test_x, test_y)
test_loader = td.DataLoader(test_ds, batch_size=64,shuffle=True)


########################################################################################################################
###DEFINE NEURAL NETWORK
########################################################################################################################
# Number of hidden layer nodes
no_features = X_train.shape[1]
no_classes = 10

# Define neural network
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(no_features, 160)
        self.fc2 = nn.Linear(160, 40)
        self.fc3 = nn.Linear(40, no_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x

model = MnistNet()
print(model)

########################################################################################################################
#TRAIN MODEL
def train_model(model, data_loader, optimizer):
    #Set model to training mode
    model.train()
    train_loss = 0
    for batch, tensor in enumerate(data_loader):
        data, target = tensor

        #feedforward
        optimizer.zero_grad()
        out = model(data)
        loss = loss_criteria(out, target)
        train_loss += loss.item()

        #backpropagate
        loss.backward()
        optimizer.step()

    avg_loss = train_loss/(batch+1)
    return avg_loss

def test_model(model, data_loader):
    #Set model to eval mode
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        batch_count = 0
        for batch, tensor in enumerate(data_loader):
            batch_count += 1
            data, target = tensor

            #Get predictions
            out = model(data)

            #update test_loss
            test_loss += loss_criteria(out, target).item()
            #Calculate the accuracy
            _, predicted = torch.max(out.data,1)
            correct += torch.sum(predicted == target)
        #Calculate loss and accuracy for test data
        avg_loss = test_loss/batch_count
        print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(avg_loss,
                                                                                         correct,
                                                                                         len(data_loader.dataset),
                                                                                         100. * correct / len(
                                                                                             data_loader.dataset)))

        return avg_loss

#Specify loss function and optimizer
loss_criteria = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer.zero_grad()

#Track training metrics
epoch_nums = []
training_loss = []
validation_loss = []

epochs = 20

for epoch in range(1, epochs+1):
    print('Epoch: {}'.format(epoch))
    train_loss = train_model(model, train_loader, optimizer)
    test_loss = test_model(model, test_loader)

    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)

plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc = 'best')
plt.show()

########################################################################################################################
#EVALUATION
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import  numpy as np

#Set model to evaluation mode
model.eval()
_, preds = torch.max(model(test_x).data,1)
fig, ax = plt.subplots(figsize = (8,8))
ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
plt.show()


print('End')