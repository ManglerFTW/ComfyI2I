# ComfyI2I

  A set of custom nodes to perform image 2 image functions in ComfyUI.
  
  If you find these nodes useful please consider contributing to it's further development. All donations are much appreciated. [Buy me a coffee](https://bmc.link/ManglerFTW)

## Installation
  If you're running on Linux, or non-admin account on windows you'll want to ensure /ComfyUI/custom_nodes, ComfyUI_I2I, and ComfyI2I.py has write permissions.
  
  There is an install.bat you can run to install to portable if detected. Otherwise it will default to system and assume you followed ComfyUI's manual installation steps.
  
  Navigate to your /ComfyUI/custom_nodes/ folder
  Run git clone https://github.com/ManglerFTW/ComfyI2I/
  Navigate to your ComfyUI_I2I folder
  Run pip install -r requirements.txt
  Start ComfyUI
  Tools will be located in the I2I menu.

## Features:

### Mask Ops Node
The mask ops node performs various mask operations on a mask created either from an image or a text prompt.

### Variables:
#### Text:
Type a prompt to create a mask from your image (make sure use_text is set to 1)

#### Text_Sigma:
The sigma factor can smooth out a mask that has been created by text. The model being used is clipseg and it might not always come out perfectly from the start. You can sometimes adjust sigma to smooth errors.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/Use_Text_Sigma 1.JPG?raw=true" width="600" alt="Sigma 0">
<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/Use_Text_Sigma 2.JPG?raw=true" width="600" alt="Sigma 90">

#### Use_Text:
0 to input a mask and 1 to use a text prompt.

#### Blend_Percentage:
You can adjust this parameter to blend your solid mask with a black and white image of what's underneath it.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/Blend_Percentage.JPG?raw=true" width="600" alt="Blend Percentage">

#### Black Level, Mid Level, White Level:
Adjust these settings to change the levels of your mask.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/Blend_Contrast.JPG?raw=true" width="600" alt="Levels">

#### Channel:
Affect the red, green, or blue channel of the underlying image.

#### Shrink_Grow:
Shrink or grow your mask using these settings.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/ShrinkGrow.JPG?raw=true" width="600" alt="Shrink_Grow">

#### Invert:
Invert your mask.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/Invert.JPG?raw=true" width="600" alt="Invert">

#### Blur_Radius:
Blur your mask.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/Blur.JPG?raw=true" width="600" alt="Blur_Radius">

### Inpaint Segments Node
This node essentially will segment and crop your mask and your image based on the mapped bounding boxes of each mask and then upscale them to 1024x1024, or a custom size of your choice. The images then go to a VAE Encode node to be processed.

 <img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/Inpaint_Segments.JPG?raw=true" width="600" alt="Inpaint Segments">

### Combine and Paste Node
The combine and paste node will take the new images from the VAE Decode node, resize them to the bounding boxes of your mask and paste them over the original image. Use color_xfer_factor to adjust the effects of the color transfer.

 <img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/Combine_and_Paste.JPG?raw=true" width="600" alt="Combine and Paste">

 ### Color Transfer Node
This is a standalone node that can take the colors of one image and transfer them to another. This function is built into the Combine and Paste node, but I decided to add it in as a standalone as well.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/Color_Transfer.JPG?raw=true" width="600" alt="Color Transfer">

 ### Workflow
 A Basic workflow with all of the nodes combined has been included in the workflows directory under I2I worflow.json. Use this as a reference to see how they are all connected.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/Workflow.JPG?raw=true" width="600" alt="Workflow">

