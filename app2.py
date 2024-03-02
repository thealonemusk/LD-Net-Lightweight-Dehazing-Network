import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Dehazer class
class Dehazer():
    def __init__(self, IMG_SIZE, LABEL_DIR, LABEL_NAME):
        self.IMG_SIZE = IMG_SIZE
        self.LABEL_DIR = LABEL_DIR
        self.LABEL_NAME = LABEL_NAME
        self.training_data = []

    def make_training_data(self):
        NUM_IMAGES = len(os.listdir(self.LABEL_DIR))
        for i in tqdm(range(1, NUM_IMAGES+1)):
            f = f"{str(i).zfill(2)}_indoor_{self.LABEL_NAME}.jpg"
            path = os.path.join(self.LABEL_DIR, f)
            if not os.path.exists(path):
                print(f"Image file at {path} does not exist.")
                continue
            img = cv2.imread(path)
            if img is None:
                print(f"Image at {path} could not be loaded.")
                continue
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
            self.training_data.append(np.array(img))
        np.save(f'{self.LABEL_NAME}.npy', self.training_data)


# Define directories
REBUILD_DATA = True
IMG_SIZE = 256
gt_dir = '../minor_project/input/GT'
hazy_dir = '../minor_project/input/hazy'


# Rebuild dataset if necessary
if REBUILD_DATA:
    dehazing_gt = Dehazer(IMG_SIZE, gt_dir, 'GT')
    dehazing_gt.make_training_data()

    dehazing_hazy = Dehazer(IMG_SIZE, hazy_dir, 'hazy')
    dehazing_hazy.make_training_data()
    
# Load dataset
patch = np.load('GT.npy', allow_pickle=True)
mask = np.load('hazy.npy', allow_pickle=True)

# Preprocess data
patch = torch.tensor(patch, dtype=torch.float32).to(device) / 255.0
mask = torch.tensor(mask, dtype=torch.float32).to(device) / 255.0

# Define model
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # batch x 32 x 256 x 256
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),   # batch x 32 x 256 x 256
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),  # batch x 64 x 256 x 256
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),  # batch x 64 x 256 x 256
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)   # batch x 64 x 128 x 128
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  # batch x 128 x 128 x 128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),  # batch x 128 x 128 x 128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),  # batch x 256 x 64 x 64
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        out = x.view(x.size(0), 256, 64, 64)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

# Create encoder and decoder instances
encoder = Encoder().to(device)
decoder = Decoder().to(device)


# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for i in range(len(patch)):
        orig_image = patch[i].unsqueeze(0).unsqueeze(0)
        hazy_image = mask[i].unsqueeze(0).unsqueeze(0)

        optimizer.zero_grad()

        encoder_output = encoder(hazy_image)
        output = decoder(encoder_output)

        loss = criterion(output, orig_image)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(patch)}], Loss: {loss.item()}')

# Save model
torch.save({
    'encoder_state_dict': encoder.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, 'dehaze_model.pth')


# Display sample images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(patch[0].cpu().squeeze(0), cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(mask[0].cpu().squeeze(0), cmap='gray')
axes[1].set_title('Hazy Image')
axes[2].imshow(output.cpu().detach().squeeze(0).squeeze(0), cmap='gray')
axes[2].set_title('Dehazed Image')
plt.show()
