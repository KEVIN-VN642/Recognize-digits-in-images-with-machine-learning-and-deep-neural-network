import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data as td
from sklearn.metrics import confusion_matrix
torch.manual_seed(1)

x_train = 'train-images.idx3-ubyte'
x_train = idx2numpy.convert_from_file(x_train).reshape(-1, 1, 28, 28)/255.0
print('X_train shape: ',x_train.shape)

y_train = 'train-labels.idx1-ubyte'
y_train = idx2numpy.convert_from_file(y_train)
print('y_train shape: ',y_train.shape)

x_test = 't10k-images.idx3-ubyte'
x_test = idx2numpy.convert_from_file(x_test).reshape(-1, 1, 28, 28)/255.0
print('X_test shape: ',x_test.shape)

y_test = 't10k-labels.idx1-ubyte'
y_test = idx2numpy.convert_from_file(y_test)
print('y_test shape: ',y_test.shape)

########################################################################################################################
#CONVERT NUMPY ARRAY TO TORCH TENSORS
train_x = torch.tensor(x_train).float()
train_y = torch.tensor(y_train).long()
train_ds = td.TensorDataset(train_x, train_y)
train_loader = td.DataLoader(train_ds, batch_size=64, shuffle=True)

test_x = torch.tensor(x_test).float()
test_y = torch.tensor(y_test).long()
test_ds = td.TensorDataset(test_x, test_y)
test_loader = td.DataLoader(test_ds, batch_size=64,shuffle=True)

########################################################################################################################
#DEFINE CNN
#Create a neural net class
num_classes = 10

class Net(nn.Module):
    #Constructor
    def __init__(self, num_classes=10):
        super(Net,self).__init__()
        # first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # max pooling with a kernel size of 2:
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        # third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16,kernel_size=3,stride=1,padding=1)
        # fully connected layer
        self.fc = nn.Linear(in_features=13*13*16, out_features=num_classes)

    def forward(self,x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # only drop the features if this is a training pass
        x = F.dropout(x,training=self.training, p=0.25)
        # flatten
        x= x.view(-1,13*13*16)
        # feed to fully-connected layer to predict classes
        x = self.fc(x)

        return F.softmax(x, dim=1)
print('CNN model class defined')

def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    train_correct = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # use the CPU or GPU as appropriate
        data, target = data.to(device), target.to(device)

        # reset the optimizer
        optimizer.zero_grad()

        # push the data forward through the model layers
        output = model(data)
        _, train_predicted = torch.max(output.data, 1)
        train_correct += torch.sum(target == train_predicted).item()

        # get the loss
        loss = loss_criteria(output, target)

        # keep a running total
        train_loss += loss.item()

        # backpropagate
        loss.backward()
        optimizer.step()

        # print metrics for every 10 batches so we see some progress
        if batch_idx % 100 == 0:
            print('Training set [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx + 1)
    print('Training set: Average loss: {:.6f} ,Accuracy {:.1f}%'.format(avg_loss, 100. *train_correct/len(train_loader.dataset)))
    return avg_loss

def test(model, device, test_loader):
    # switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)

            # Get the predicted classes for this batch
            output = model(data)

            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()

            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # return average loss for the epoch
    return avg_loss


# identify device
device = "cpu"
if (torch.cuda.is_available()):
    # if GPU available, use cuda (on a cpu, training will take a considerable length of time!)
    device = "cuda"
print('Training on', device)

model = Net(num_classes).to(device)
# set up optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.0005)
# specify loss criteria
loss_criteria = nn.CrossEntropyLoss()

# Train the model
epoch_nums = []
training_loss = []
validation_loss = []
epochs = 15
for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)

def loss_chart():
    plt.plot(epoch_nums, training_loss)
    plt.plot(epoch_nums, validation_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()

def evaluate_model():
    model.eval()
    print('Getting predictions from test set...')
    truelabels = []
    predictions = []
    for data, target in test_loader:
        for label in target.cpu().data.numpy():
            truelabels.append(label)
        for pred in model.cpu()(data).data.numpy().argmax(1):
            predictions.append(pred)
    # Plot the confusion matrix
    cm = confusion_matrix(truelabels, predictions)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel("Predicted Shape")
    plt.ylabel("Actual Shape")
    plt.show()


# Save the model weights
model_file = 'models/mnist_cnn_torch.pt'
torch.save(model.state_dict(), model_file)
del model
print('model saved as', model_file)

print('End')