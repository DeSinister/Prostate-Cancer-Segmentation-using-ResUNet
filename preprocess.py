from PIL import Image
import pandas as pd
import torch
import numpy as np
import glob
import SimpleITK as sitk
from torchvision import transforms

# Reference Spacing for Uniform Spatial Referencing the 3D Images
ref_spacing = [0.5, 0.5, 3.0]

def resample_image(img, ref_spacing, label = False):
    # Store Input Image stats
    input_size = img.GetSize()
    input_spacing = img.GetSpacing()
    input_direction = img.GetDirection()
    input_origin = img.GetOrigin()
    
    # Output Size is changed to adjust the change in Output Spacing, so the Anatomy doesnt change 
    output_size = [round(input_size[j] * (input_spacing[j] / ref_spacing[j])) for j in range(3)]
    
    # Initialise Resample Filter
    resampler = sitk.ResampleImageFilter()

    # Setting Output Stats for the Resample Filter
    resampler.SetSize(output_size)
    resampler.SetOutputSpacing(ref_spacing)
    resampler.SetOutputDirection(input_direction)
    resampler.SetOutputOrigin(input_origin)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(img.GetPixelIDValue())

    # Using BSpline Interploation Method for Input Image and nearest-Neighbour for Labels
    if label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
    
    # Convert Image to Array Format
    img = sitk.GetArrayFromImage(resampler.Execute(img))
    z, x, y = img.shape
    
    # Slicing Images to return middle 8 slices, containing 300x300px middle copped content
    img = img[z//2 - 8: z//2 + 8, x//2-150: x//2 + 150,  y//2 - 150: y//2 + 150]
    
    return img

# Function to normalize the Input Images by Standardization
def norm(img):
    # Calculating mean and Standard Deviation
    m = np.mean(img)
    s = np.std(img)

    # Standard Practice to replace denominator by 1 if it is 0 to avoid division by zero error
    if s ==0:
        s=1
    # Subtract mean and divide by standard deviation
    img = (img-m)/s

    # convert normalized 0-1 values to 255 scale
    img = img*255

    # convert the numpy array image to a PIL Image
    img = Image.fromarray(img).convert("L")
    return img


# Function to Binarize the Label maps to 0 or 1 value for the UNET to use Binary classification for Each Pixel
def thresh(img):
    # Changing Permission of Image Format to be Editable
    img = img.copy()
    img.setflags(write=1)
    
    # Converting Every value greater than 0 to 1 for Binarizing the Imaging
    img[img>0] = 1

    # converting 0-1 Normalized scale to 255 scale
    img = img*255
    img = img.astype(np.uint8)

    # convert the numpy array image to a PIL Image
    img = Image.fromarray(img).convert("1")
    return img


# An Encapsuled class which taskes in the folds as location and stores the preprocessed files to their respective splits
class PrepFiles():

    # Constructor to Initialize the folds
    def __init__(self, f):
        self.f = f
    
    def prepare_fold(self, split):
        # for Everty Fold in the Split Stratified
        for i in self.f:
            # Generalized Input Path
            path = "C:\\Users\\nishq\\Downloads\\picai\\"

            # For Every File Containing in the Respective Fold with extension t2w.mha
            for j in glob.glob(path + f"picai_public_images_fold{i}\\*\\*t2w.mha"):
                # Reading the t2w Files
                t2w = resample_image(sitk.ReadImage(j), ref_spacing)
                
                # Considering the Respective ID for The T2W Image and using it to extract the other Images
                adc = resample_image(sitk.ReadImage(j[:-7]+"adc.mha"), ref_spacing)
                hbv = resample_image(sitk.ReadImage(j[:-7]+"hbv.mha"), ref_spacing)
                lesion = resample_image(sitk.ReadImage(path + f"picai_labels-main\\csPCa_lesion_delineations\\AI\\Bosma22a\\{j[63:-8]}.nii.gz"), ref_spacing, True)
                whole_gland = resample_image(sitk.ReadImage(path + f"picai_labels-main\\anatomical_delineations\\whole_gland\\AI\\Bosma22b\\{j[63:-8]}.nii.gz"), ref_spacing, True)
                
                # As we have 16 Different Slices, the Iterator is used to process each 2D Slice
                for k in range(16):
                        # Extracting ID that is {Study}_{Patient}
                        id = j[63:-8] 

                        # Extracting kth slice of 16 2D slices
                        # Normalizing the Input Images
                        t2w_img = norm(t2w[k])
                        adc_img = norm(adc[k])
                        hbv_img = norm(hbv[k])

                        # Thresholding the Labels
                        les_img = thresh(lesion[k])
                        gld_img = thresh(whole_gland[k])
                        
                        # Saving The Images, so The Images are not required to preprocess all the time
                        t2w_img.save(path + "main2\\" + split + f"\\{id}_{k}_o_t2w.jpg")
                        adc_img.save(path + "main2\\" + split + f"\\{id}_{k}_o_adc.jpg")
                        hbv_img.save(path + "main2\\" + split + f"\\{id}_{k}_o_hbv.jpg")
                        les_img.save(path + "main2\\" + split + f"\\{id}_{k}_o_les.jpg")
                        gld_img.save(path + "main2\\" + split + f"\\{id}_{k}_o_gld.jpg")
                        
                        # Horizontal Flip Augmentation is applied to Each repective Image and are saved for increasing the Size of the Data
                        t2w_img = transforms.functional.hflip(t2w_img)
                        t2w_img.save(path + "main2\\" + split + f"\\{id}_{k}_f_t2w.jpg")
                        
                        adc_img = transforms.functional.hflip(adc_img)
                        adc_img.save(path + "main2\\" + split + f"\\{id}_{k}_f_adc.jpg")
                        
                        hbv_img = transforms.functional.hflip(hbv_img)
                        hbv_img.save(path + "main2\\" + split + f"\\{id}_{k}_f_hbv.jpg")
                        
                        les_img = transforms.functional.hflip(les_img)
                        les_img.save(path + "main2\\" + split + f"\\{id}_{k}_f_les.jpg")
                        
                        gld_img = transforms.functional.hflip(gld_img)
                        gld_img.save(path + "main2\\" + split + f"\\{id}_{k}_f_gld.jpg")

                        # The Sign convention to save the File is: {Study}_{Patient}_{2D_Slice_Number}_{Original(o)/Flip(f)}_{File_Type}.jpg


# Preparing Files for Training Data using Folds 1, 2, and 4
train = PrepFiles([1, 2, 4])
train.prepare_fold("train")
print("Train Prepared")


# Preparing Files for Training Data using Folds 1, 2, and 4
valid = PrepFiles([3])
valid.prepare_fold("valid")
print("Validation Prepared")


# Preparing Files for Training Data using Folds 1, 2, and 4
test = PrepFiles([0])
test.prepare_fold("test")
print("Test Prepared")