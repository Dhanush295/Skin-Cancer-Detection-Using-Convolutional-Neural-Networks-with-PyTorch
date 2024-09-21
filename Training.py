import os
import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

TRAIN_DIR = 'C:\\Users\\dhanu\\OneDrive\\Desktop\\DK\\train'
TEST_DIR = 'C:\\Users\\dhanu\\OneDrive\\Desktop\\DK\\test'

IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = f'skincancer-{LR}-2conv-basic.pth'

class SkinCancerDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        for img in tqdm(os.listdir(self.directory)):
            label = label_img(img)
            if label is not None:
                path = os.path.join(self.directory, img)
                img_data = cv2.imread(path, cv2.IMREAD_COLOR)
                img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
                self.data.append(np.array(img_data))
                self.labels.append(label)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0, torch.tensor(label, dtype=torch.float32)

def label_img(img):
    word_label = img[0]
    if word_label == 'b':
        return [1, 0]  # benign
    elif word_label == 'm':
        return [0, 1]  # malignant
    return None

# Load the training dataset
train_dataset = SkinCancerDataset(TRAIN_DIR)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the neural network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 1024)
        self.fc2 = nn.Linear(1024, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(nn.ReLU()(self.fc1(x)))
        x = self.fc2(x)
        return x

model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, torch.argmax(labels, dim=1))
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), MODEL_NAME)

# Process the test data similarly as you did for the training data
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

test_data = process_test_data()
