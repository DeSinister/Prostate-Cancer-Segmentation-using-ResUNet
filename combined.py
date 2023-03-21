import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pandas as pd
import torch
import numpy as np
import glob
import SimpleITK as sitk
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ------- Setting Hyperparameters -------
batch_size = 26
num_of_epochs = 50
learning_rate = 2e-6
learning_rate_patience = 5
early_stop_patience = 15
device = torch.device('cuda')
num_workers = 0

# --- Data Parameters ---
# Defining the Input Image and Label types, to avoid reading Every Image into the Mermory
input_image = "hbv"
target_image = "les"


# Class to Read and Return the SDtored Preprocess Data
class FetchData():
    # For Constructor, the Parameters It takes the Split Type, Input Image Type,  Label Image Type, Transformation Fucntions for Input Images, and Transformation Functions for Target Images
    def __init__(self, split, input_image, target_image, inpt, tart):
        self.split = split
        self.input_image = input_image
        self.target_image = target_image
        self.inpt = inpt
        self.tart = tart
    
    # Function To return the Fetched Data
    def prepare_fold(self):
        dataset = []
        path = r"C:\Users\nishq\Downloads\picai\\main2" + "\\" + self.split
        # For Every Image Type stored in the Split is read and the matched Input Image and target Image are stored in a List, which is stored in other exhaustive list
        for j in glob.glob(path + f"\\*25_8*{self.input_image}.jpg"):
            # Read the Input Image in grayscale Format as being a single channel Image, and Apply the Transformation Function
            input = self.inpt(Image.open(j).convert('L'))
            # Read Target Image in a binary Format as being just 0/1 value, and Apply The Transformation Function
            target = self.tart(Image.open(j[:-7]+f"{self.target_image}.jpg").convert('1'))
            # Append the Dataset List
            dataset.append([np.array(input), np.array(target)])
        return dataset


# ------- Data Prepration -------
# Preparing the Training Dataset with applied Tensor Transformation
train = FetchData("train", input_image, target_image, transforms.Compose([transforms.ToTensor()]), transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.to(torch.int))]))
train_dataset = train.prepare_fold()

# Preparing the Validation Dataset with applied Tensor Transformation
valid = FetchData("valid", input_image, target_image, transforms.Compose([transforms.ToTensor()]), transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.to(torch.int))]))
valid_dataset = valid.prepare_fold()

# Preparing the Testing Dataset with applied Tensor Transformation
test = FetchData("test", input_image, target_image, transforms.Compose([transforms.ToTensor()]), transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.to(torch.int))]))
test_dataset = test.prepare_fold()

# Preparing the Training Data Loaders with the Batch Size
train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

# Preparing the Validation Data Loaders with the Batch Size
valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

# Preparing the Testing Data Loaders with the Batch Size
test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


import torch
import torch.nn as nn

# Class For an Encoder Block, which Contracts the Image
class EncoderBlock(nn.Module):
    
    # The constructor takes in Number of Input channels, Number of Output channels and the Number of Strides
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        # Batch Normalization Followed by ReLU
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu1 = nn.ReLU()

        # Convolution Layer - 1
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=stride)

        # Batch Normalization Followed by ReLU
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, stride=1)

        # Convolution Layer - 2
        self.skipconv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=stride)

        # Adding Dropout to Avoid Over-fitting
        self.dropout = nn.Dropout(0.25) 

    def forward(self, input):
        # Calling Batch Normalization and ReLU
        x = self.bn1(input)
        x = self.relu1(x)
        # Dropout Layer to Avoid Overfitting to the Convolutional Layer - 1
        x = self.dropout(self.conv1(x))

        # Calling Batch Normalization and ReLU
        x = self.bn2(x)
        x = self.relu2(x)
        # Dropout Layer to Avoid Overfitting to the Convolutional Layer - 1
        x = self.dropout(self.conv2(x))

        # Skip Connection 
        s = self.skipconv1(input)

        # Residual Path is added by adding the Skip Connection
        skip1 = x + s
        return skip1
    
# Class for Decoder Block, which Expands the Image
class DecoderBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        # Intializing the Scaling Factor
        self.up = nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True)

        # Using the Residual Encoder Block But This time to Decrease the Number of Channels
        self.EncoderBlock1 = EncoderBlock(input_channel + output_channel, output_channel, 1)

        # Convolutional Layer when the Odd Number was previously Halved by floor Division and Now when If doubled is not the same as before
        self.convm = nn.Conv2d(input_channel, input_channel, kernel_size=2, padding=0)


    def forward(self, input, skip):
        # Setting Upscaling Factor
        x = self.up(input)
        
        # If the Image has Odd Number of Dimension, and Additional Convolutional Layer is used to make it the same dimension
        if int(skip.shape[-1])%2!=0:
            x = self.convm(x)
        # After being the same dimension, they can be Concatenated.
        x = torch.cat([x, skip], axis=1)

        # Encoder Block the Narrow the Number of Channels
        x = self.EncoderBlock1(x)
        
        return x

# 4 - Layer Architecture for Residual UNET
class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Encoder Block - 1 ---- 
        # 300, 300, 1 -> 300, 300, 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding = 1) 
        # Batch Normalization Followed by ReLU
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        # 300, 300, 64 -> 300, 300, 64
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # 300, 300, 1  -> 300, 300, 64
        self.skipconv1 = nn.Conv2d(1, 64, kernel_size=1, padding=0)

        # ---- Encoder Block - 2 ---- 
        # 300, 300, 64 -> 150, 150, 128
        self.encoderblock2 = EncoderBlock(64, 128, 2)

        # ---- Encoder Block - 3 ---- 
        # 150, 150, 128 -> 75, 75, 256
        self.encoderblock3 = EncoderBlock(128, 256, 2)

        # ---- Encoder Block - 4 ---- 
        # 75, 75, 256 -> 38, 38, 512
        self.encoderblock4 = EncoderBlock(256, 512, 2)

        # ---- Bottleneck Block ---- 
        # 38, 38, 512 -> 19, 19, 1024
        self.encoderblock5 = EncoderBlock(512, 1024, 2)

        # ---- Decoder Block - 4 ---- 
        # 19, 19, 1024 -> 38, 38, 512
        self.decoderblock4 = DecoderBlock(1024, 512)
        
        # ---- Decoder Block - 3 ---- 
        # 38, 38, 512 -> 76, 76, 256
        self.decoderblock3 = DecoderBlock(512, 256)
        # 76, 76, 512 -> 75, 75, 512

        # ---- Decoder Block - 2 ---- 
        # 75, 75, 256 -> 150, 150, 128
        self.decoderblock2 = DecoderBlock(256, 128)

        # ---- Decoder Block - 1 ---- 
        # 150, 150, 128 -> 300, 300, 64
        self.decoderblock1 = DecoderBlock(128, 64)

        # ---- Merging Channels ---- 
        # 300, 300, 64 -> 300, 300, 1
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        # Sigmoid Function to Convert the Values to [0, 1]
        self.sigmoid = nn.Sigmoid()

        # Dropout Layer to Avoid Overfitting
        self.dropout = nn.Dropout(0.5) 


    def forward(self, input):
        # Encoder Block - 1
        # 300, 300, 1 -> 300, 300, 64
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        s = self.skipconv1(input)
        skip1 = x + s

        # Encoder Block - 2
        # 300, 300, 64 -> 150, 150, 128
        skip2 = self.encoderblock2(skip1)

        # Encoder Block - 3
        # 150, 150, 128 -> 75, 75, 256
        skip3 = self.encoderblock3(skip2)

        # Encoder Block - 4
        # 75, 75, 256 -> 38, 38, 512
        skip4 = self.encoderblock4(skip3)

        # Bottle Neck
        # 38, 38, 512 -> 19, 19, 1024
        skip5 = self.encoderblock5(skip4)

        # Decoder Block - 4
        # 19, 19, 1024 -> 38, 38, 512
        skip6 = self.decoderblock4(skip5, skip4)

        # Decoder Block - 3
        # 38, 38, 512 -> 75, 75, 256
        skip7 = self.decoderblock3(skip6, skip3)

        # Decoder Block - 2
        # 75, 75, 256 -> 150, 150, 128
        skip8 = self.decoderblock2(skip7, skip2)

        # Decoder Block - 1
        # 150, 150, 128 -> 300, 300, 64
        skip9 = self.dropout(self.decoderblock1(skip8, skip1))

        # Merge
        # 300, 300, 64 -> 300, 300, 1
        x = self.conv3(skip9)
        x = self.sigmoid(x)

        return x


# 3 - Layer Architecture for Residual UNET
class MiniResUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Encoder Block - 1 ---- 
        # 300, 300, 1 -> 300, 300, 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding = 1) 
        # Batch Normalization Followed by ReLU
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        # 300, 300, 64 -> 300, 300, 64
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # 300, 300, 1  -> 300, 300, 64
        self.skipconv1 = nn.Conv2d(1, 64, kernel_size=1, padding=0)

        # ---- Encoder Block - 2 ---- 
        # 300, 300, 64 -> 150, 150, 128
        self.encoderblock2 = EncoderBlock(64, 128, 2)

        # ---- Encoder Block - 3 ---- 
        # 150, 150, 128 -> 75, 75, 256
        self.encoderblock3 = EncoderBlock(128, 256, 2)

        # ---- Bottleneck Block ---- 
        # 75, 75, 256 -> 38, 38, 512
        self.encoderblock4 = EncoderBlock(256, 512, 2)
        
        # ---- Decoder Block - 3 ---- 
        # 38, 38, 512 -> 76, 76, 256
        self.decoderblock3 = DecoderBlock(512, 256)
        # 76, 76, 512 -> 75, 75, 512

        # ---- Decoder Block - 2 ---- 
        # 75, 75, 256 -> 150, 150, 128
        self.decoderblock2 = DecoderBlock(256, 128)

        # ---- Decoder Block - 1 ---- 
        # 150, 150, 128 -> 300, 300, 64
        self.decoderblock1 = DecoderBlock(128, 64)

        # ---- Merging Channels ---- 
        # 300, 300, 64 -> 300, 300, 1
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        # Sigmoid Function to Convert the Values to [0, 1]
        self.sigmoid = nn.Sigmoid()

        # Dropout Layer to Avoid Overfitting
        self.dropout = nn.Dropout(0.5) 


    def forward(self, input):
        # Encoder Block - 1
        # 300, 300, 1 -> 300, 300, 64
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        s = self.skipconv1(input)
        skip1 = x + s

        # Encoder Block - 2
        # 300, 300, 64 -> 150, 150, 128
        skip2 = self.encoderblock2(skip1)

        # Encoder Block - 3
        # 150, 150, 128 -> 75, 75, 256
        skip3 = self.encoderblock3(skip2)

        # Bottle Neck
        # 75, 75, 256 -> 38, 38, 512
        skip4 = self.encoderblock4(skip3)

        # Decoder Block - 3
        # 38, 38, 512 -> 75, 75, 256
        skip7 = self.decoderblock3(skip4, skip3)

        # Decoder Block - 2
        # 75, 75, 256 -> 150, 150, 128
        skip8 = self.decoderblock2(skip7, skip2)

        # Decoder Block - 1
        # 150, 150, 128 -> 300, 300, 64
        skip9 = self.dropout(self.decoderblock1(skip8, skip1))

        # Merge
        # 300, 300, 64 -> 300, 300, 1
        x = self.conv3(skip9)
        x = self.sigmoid(x)

        return x


# 2 - Layer Architecture for Residual UNET
class TinyResUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Encoder Block - 1 ---- 
        # 300, 300, 1 -> 300, 300, 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding = 1) 
        # Batch Normalization Followed by ReLU
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        # 300, 300, 64 -> 300, 300, 64
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # 300, 300, 1  -> 300, 300, 64
        self.skipconv1 = nn.Conv2d(1, 64, kernel_size=1, padding=0)

        # ---- Encoder Block - 2 ---- 
        # 300, 300, 64 -> 150, 150, 128
        self.encoderblock2 = EncoderBlock(64, 128, 2)

        # ---- BottleNeck Block ---- 
        # 150, 150, 128 -> 75, 75, 256
        self.encoderblock3 = EncoderBlock(128, 256, 2)

        # ---- Decoder Block - 2 ---- 
        # 75, 75, 256 -> 150, 150, 128
        self.decoderblock2 = DecoderBlock(256, 128)

        # ---- Decoder Block - 1 ---- 
        # 150, 150, 128 -> 300, 300, 64
        self.decoderblock1 = DecoderBlock(128, 64)

        # ---- Merging Channels ---- 
        # 300, 300, 64 -> 300, 300, 1
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        # Sigmoid Function to Convert the Values to [0, 1]
        self.sigmoid = nn.Sigmoid()

        # Dropout Layer to Avoid Overfitting
        self.dropout = nn.Dropout(0.5) 


    def forward(self, input):
        # Encoder Block - 1
        # 300, 300, 1 -> 300, 300, 64
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        s = self.skipconv1(input)
        skip1 = x + s

        # Encoder Block - 2
        # 300, 300, 64 -> 150, 150, 128
        skip2 = self.encoderblock2(skip1)

        # Bottle Neck
        # 150, 150, 128 -> 75, 75, 256
        skip3 = self.encoderblock3(skip2)

        # Decoder Block - 2
        # 75, 75, 256 -> 150, 150, 128
        skip8 = self.decoderblock2(skip3, skip2)

        # Decoder Block - 1
        # 150, 150, 128 -> 300, 300, 64
        skip9 = self.dropout(self.decoderblock1(skip8, skip1))

        # Merge
        # 300, 300, 64 -> 300, 300, 1
        x = self.conv3(skip9)
        x = self.sigmoid(x)

        return x


# --- Model Configurations ---
model = ResUNet()
model.to(device)
scaler = torch.cuda.amp.GradScaler()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=learning_rate_patience, verbose=True)



# Fucntion to Calculate the Duration of an epoch
def epoch_time(start_time, end_time):
    # End Time - Start Time
    duration = end_time - start_time

    # Convert into Minutes and Seconds
    minutes = duration//60
    seconds = duration - (60*minutes)

    return int(minutes), int(seconds)


# Function to Calculate the Dice Score
def dice_score(prediction, label, smooth_parameter = 1):
    # Memory Tensor Conatining Contiguous values
    label = label.contiguous()
    prediction = prediction.contiguous()    
    
    # Intersection Calculated By Multiplying the 2 Tensors, as the Task is Binary Classification
    intersection = (prediction * label).sum(dim=2).sum(dim=2)

    # Calculating Dice Score as Intersection Area / (Predicted Area + Label Area)
    score =  (2. * intersection + smooth_parameter) / (prediction.sum(dim=2).sum(dim=2) + label.sum(dim=2).sum(dim=2) + smooth_parameter)
    
    # Return Average mean
    return score.mean() 


# Function to Calculate the Dice Loss
def dice_loss(prediction, label, smooth_parameter = 1):
    # Memory Tensor Conatining Contiguous values
    label = label.contiguous()
    prediction = prediction.contiguous()    
    
    # Intersection Calculated By Multiplying the 2 Tensors, as the Task is Binary Classification
    intersection = (prediction * label).sum(dim=2).sum(dim=2)

    # Calculating Dice Loss = 1 - Dice Score; Where, Dice Score is Intersection Area / (Predicted Area + Label Area)
    loss =  (1 - ((2. * intersection + smooth_parameter) / (prediction.sum(dim=2).sum(dim=2) + label.sum(dim=2).sum(dim=2) + smooth_parameter))) 
    
    # Return Average mean
    return loss.mean() 


# Function to Train the Model for Every Epoch
def train_model(model, loader, optimizer, loss_fn, device, scaler):
    # Intializing starting Epoch loss as 0
    loss_for_epoch = 0.0

    # Model to be used in Training Mode
    model.train()

    # For every Input Image, Label Image in a Batch
    for x, y in loader:

        # Storing the Images to the Device
        x = x.to(device, dtype=torch.float16)
        y = y.to(device, dtype=torch.float16)

        # Set Gradient of all parameters to 0
        optimizer.zero_grad()
        
        # Using Unscaled Mixed Precision using half Bit for Faster Processing
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # Get Predictions from Model
            y_pred = model(x)

            # Calculate the Loss
            loss = loss_fn(y_pred, y)

        # Scale Loss Backwards
        scaler.scale(loss).backward()

        # Unscale the Gradients in Optimizer
        scaler.unscale_(optimizer)

        # Clip the Gradients to they dontreach inf
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        
        # Update the Scaler
        scaler.update()

        # Add the Loss for every sample in a Batch
        loss_for_epoch += loss.item()

    # Calculating The Average Loss for the Epoch
    loss_for_epoch =  torch.div(loss_for_epoch, len(loader))
    return loss


# Function to Evaluate the Model
def evaluate_model(model, loader, loss_fn, device):
    # Intializing starting Epoch loss as 0
    total_loss = 0.0
    
    # Model to be used in Evaluation Mode
    model.eval()

    # Gradients are not calculated
    with torch.no_grad():
        
        # For every Input Image, Label Image in a Batch
        for x, y in loader:

            # Storing the Images to the Device
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            # Get Predictions from Model
            y_pred = model(x)

            # Calculate the Loss
            loss = loss_fn(y_pred, y)
            
            # Add the Loss for every sample in a Batch
            total_loss += loss

        # Calculating The Average Loss for the Epoch
        total_loss =  torch.div(total_loss, len(loader))
        
    return total_loss




# ------- Training the Model -------
# Training Initialization Setup
lowest_loss = 1.0
training_loss = []
validation_loss = []
# Variable Setup for Early Stopping
bad_epochs = 0
early_stop = False
end_epoch = num_of_epochs
# Loop for Every epochs
for epoch in range(num_of_epochs):
    # Start Counting time
    start = time.time()

    # Train the Model for Every epoch
    training_loss.append(train_model(model, train_loader, optimizer, dice_loss, device, scaler))

    # Evaluate the Model using the Validation Split
    validation_loss.append(evaluate_model(model, valid_loader, dice_loss, device))
    
    # Save the Model If the Model is Performing better on Validation set while Training
    if validation_loss[-1] < lowest_loss:
        # Reset Patience for Early Stopping
        bad_epochs = 0
        print(f"Validation Loss Decreased from {lowest_loss} to {validation_loss[-1]}")
        # Changing the Lowest Loss to Current Validation Loss
        lowest_loss = validation_loss[-1]
        # Saving the Model
        torch.save(model.state_dict(), rf"C:\Users\nishq\OneDrive\Desktop\{input_image}_to_{target_image}epoch_{epoch}_loss{validation_loss[-1]}.pt")
    else:
        # Model not performning better for this epoch
        bad_epochs+=1

    # Stop the Counting Time
    end = time.time()

    # Estimate the Duration
    minutes, seconds = epoch_time(start, end)

    # Report Training and Validation Loss
    print(f"Epoch Number: {epoch+1}")
    print(f"Duration: {minutes}m {seconds}s")
    print(f"Training Loss: {training_loss[-1]}")
    print(f"Validation Loss: {validation_loss[-1]}")
    print()


    # If Patience Level reached for Model not Performing better
    if bad_epochs == early_stop_patience:
        print("Stopped Early. The Model is not improving over validation loss")
        end_epoch = epoch
        break



# ------- Evaluating the Model -------
# Calculating Testing Loss
training_score = evaluate_model(model, train_loader, dice_score, device).cpu()
validation_score = evaluate_model(model, valid_loader, dice_score, device).cpu()
test_score = evaluate_model(model, test_loader, dice_score, device).cpu()
print("Testing Dice Score: ", test_score)
# Calculating the Average Loss and Average Weighted Loss
print("Average Loss: ", (training_score + validation_score + test_score)/3)
print("Average Weighted Loss: ", (3*training_score + validation_score + test_score)/5)


# --- Visualizing Results ---
with torch.no_grad():
    # Extracting Loss Values
    training_loss = [x.item() for x in training_loss]
    validation_loss = [x.item() for x in validation_loss]
# Plotting Training Loss with Green color
plt.plot([x for x in range(1, end_epoch+1)], training_loss, 'g', label='Training Dice loss')
# Plotting Validation Loss with Blue Color
plt.plot([x for x in range(1, end_epoch+1)], validation_loss, 'b', label='Validation Dice loss')
# Setting Title
plt.title('Training and Validation Loss')
# Setting X and Y Labels
plt.xlabel('Number of Epochs')
plt.ylabel('Dice Loss')
plt.legend()
plt.show()