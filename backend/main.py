
import io
import os
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.autograd import Variable

import matplotlib.pyplot as plt

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

# Define the directory to store the uploaded images
UPLOAD_DIRECTORY = "uploads"

# Create the uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,32,3,padding=1),   # batch x 32 x 256 x 256
                        nn.ReLU(),
                        nn.BatchNorm2d(32),             
                        nn.Conv2d(32,32,3,padding=1),   # batch x 32 x 256 x 256
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,64,3,padding=1),  # batch x 64 x 256 x 256
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64,64,3,padding=1),  # batch x 64 x 256 x 256
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(2,2)   # batch x 64 x 128 x 128
        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,3,padding=1),  # batch x 128 x 128 x 128
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.Conv2d(128,128,3,padding=1),  # batch x 128 x 128 x 128
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(128,256,3,padding=1),  # batch x 256 x 64 x 64
                        nn.ReLU()
        )
                
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(1, -1)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.ConvTranspose2d(256,128,3,2,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128,128,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128,64,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.ConvTranspose2d(64,64,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
                        nn.ConvTranspose2d(64,32,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32,32,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32,1,3,2,1,1),
                        nn.ReLU()
        )
        
    def forward(self,x):
        out = x.view(1,256,64,64)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/image-upload")
async def dehaze_image(image: UploadFile = File(...)):
    encoder = Encoder()
    decoder = Decoder()
    try:
        file_path = os.path.join(UPLOAD_DIRECTORY, image.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await image.read())

        # Load the image using PIL
        uploaded_image = Image.open(file_path)
            
        image_np = np.array(uploaded_image)

        # Convert the NumPy array to BGR (OpenCV format)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        #for debug purpose only
        # image.show()
        # Define the preprocessing steps
        img=cv2.resize(image_bgr,(256,256))

        X_orig = torch.Tensor(img)
        X_orig=X_orig/255

        X_orig = X_orig.unsqueeze(0)
        X_orig_T = torch.transpose(X_orig, 1, 3)

        input_image=X_orig_T.reshape(-1,1,256,256)
        print("Size of X_orig_T:", input_image.shape)

        hazy_loader = torch.utils.data.DataLoader(dataset=input_image, batch_size=1)

        # Iterate over batches of hazy images (only one iteration for a single image)
        for hazy_image in hazy_loader:
            # Assuming you're using Variable from torch.autograd
            # Convert to Variable (if necessary) and move to GPU
            hazy_image = hazy_image

        #load model
        try:
            model = torch.load(r'C:\Users\agrim\OneDrive\Desktop\minor-project\backend1\dehaze_autoencoder.pkl', map_location=torch.device('cpu'))
            encoder = model[0]
            print("Encoder------------<><><><><>-----------",encoder)
            decoder = model[1]
        except Exception as e:
            raise Exception(400, str(e))
        

        train_hazy_loader = torch.utils.data.DataLoader(dataset=input_image, batch_size=1, shuffle=False)

        dehazed_output = []

        # Iterate over batches of hazy images (only one iteration for a single image)
        for train_hazy in tqdm(train_hazy_loader):
            hazy_image = Variable(train_hazy)

            # Forward pass through encoder and decoder
            encoder_op = encoder(hazy_image)
            output = decoder(encoder_op)

            # Detach output and move to CPU
            output = output.cpu().detach()

            # Append output to list
            dehazed_output.append(output)

        # Convert list of outputs to tensor
        X_dehazed = torch.stack(dehazed_output)

        # Reshape tensor to match desired shape
        X_dehazed = X_dehazed.view(-1, 1, 256, 256)  # Assuming single channel
        X_dehazed = X_dehazed.view(-1, 3, 256, 256)  # Convert to 3 channels
        X_dehazed = X_dehazed.permute(0, 2, 3, 1)     # Permute dimensions to match expected format (batch_size, height, width, channels)

        print("X_DEHAZE--------------><><><><><><><>-------------------------",X_dehazed) 

        # Plot the dehazed image
        rotated_image = np.rot90(X_dehazed.squeeze(), k=-1) 
        mirror_image_horizontal = np.flip(rotated_image, axis=1)

        mirror_image_horizontal = (mirror_image_horizontal * 255).clip(0, 255).astype(np.uint8)
        # Return the output image as a response stream
        # Convert the image array to bytes
        # Save the image to a temporary file
        temp_file_path = "uploads/temp_one_image.jpg"
        cv2.imwrite(temp_file_path, mirror_image_horizontal)

        # Return the image file as a response
        return FileResponse(temp_file_path, media_type="image/jpeg")
    
    except Exception as e:
        raise Exception(400, str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)