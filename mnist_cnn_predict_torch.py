import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.image as mpimg
import os
import random

print("Libraries imported - ready to use PyTorch", torch.__version__)
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

# function to predict the class of an image
def predict_image(classifier, image):
    """
    :param classifier: the trained model
    :param image: image need to classify
    :return: label of image
    """
    # set model to evaluation mode
    classifier.eval()
    # reading image
    im = mpimg.imread(path+image)
    # the model expects a batch of images as input, so we'll create an array of 1 image
    im = im.reshape(1, 1, im.shape[0], im.shape[1])/255.0
    # We need to format the input to match the training data
    im = torch.tensor(im).float()
    # Use the model to predict the image class
    output = classifier(im)
    # Find the class predictions with the highest predicted probability
    index = output.data.numpy().argmax()
    return index

# identify device
device = "cpu"
if (torch.cuda.is_available()):
    # if GPU available, use cuda (on a cpu, training will take a considerable length of time!)
    device = "cuda"
print('Training on', device)
# path to trained model
model_file = 'models/mnist_cnn_torch.pt'
model = Net().to(device)
# load model
model.load_state_dict(torch.load(model_file))
print(model)
# path to test_images folder
path = 'test_images/'

def visual_prediction():
    """
    this function visualize 4 randomly selected images and their predicted labels
    :return:
    """
    # get list of test images
    img_list = os.listdir(path)
    # get random 4 images to display
    test_imgs = random.sample(img_list,4)

    fig = plt.figure(figsize=(8, 6))
    # visualize predicted labels and actual images
    i=0
    for im in test_imgs:
        i += 1
        a = fig.add_subplot(2,2,i)
        a.axis('off')
        label = predict_image(model,im)
        im = mpimg.imread(path+im)
        plt.imshow(im)
        a.set_title('Predicted label: '+ str(label))
    plt.show()

visual_prediction()

print('End')





