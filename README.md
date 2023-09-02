# ComfyI2I

  A set of custom nodes to perform image 2 image functions in ComfyUI.
  
  If you find these nodes useful please consider contributing to it's further development. All donations are much appreciated. [Buy me a coffee](https://bmc.link/ManglerFTW)
  
## V2 Update - 9/2/2023

## New Features
- Mask_Ops node will now output the whole image if mask = None and use_text = 0
- Mask_Ops node now has a separate_mask function that if 0, will keep all mask islands in 1 image vs separating them into their own images if it's at 1 (use 0 for color transfer)
- Significantly improved Color_Transfer node
  - Extract up to 256 colors from each image (generally between 5-20 is fine) then segment the source image by the extracted palette and replace the colors in each segment
  - Set a blur to the segments created
  - Control the strength of the color transfer function
  - Controls for Gamma, Contrast, and Brightness

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

### Color Transfer Node (improved for V2)
This is a standalone node that can take the colors of one image and transfer them to another.

### Variables:
#### No_of_Colors:
Choose the amount of colors you would like to extract from each image. Usually beteen 5-20 is fine. For smaller masked regions you can bring it to 1-5 and it will only extract those that are most dominant.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/V2/no_of_colors.JPG?raw=true" width="600" alt="no_of_colors">

#### Blur_Radius and Blur_Amount:
These controls will effect how the edges of the separated color segments in the source image blur at their respective edges.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/V2/blur_radius.JPG?raw=true" width="600" alt="blur_radius">

#### Strength:
Adjust Strength to control how weak or strong you would like the color transfer effect to be.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/V2/strength.JPG?raw=true" width="600" alt="strength">

#### Gamma:
Adjust Gamma to control the Gamma amount of the resulting image.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/V2/gamma.JPG?raw=true" width="600" alt="gamma">

#### Contrast:
Adjust Contrast to control the Contrast amount of the resulting image.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/V2/contrast.JPG?raw=true" width="600" alt="contrast">

#### Brightness:
Adjust Brightness to control the Brightness amount of the resulting image.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/V2/brightness.JPG?raw=true" width="600" alt="brightness">

#### Masked Color Transfer:
The Color Transfer node now works with masked regions giving you more control with which areas to transfer color to.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/V2/masked_xfer.JPG?raw=true" width="600" alt="masked_xfer">

#### Interoperability with the Mask_Ops node:
The Color Transfer node works well with masks created with the Mask_Ops node. The example below shows color transfer difference across each individial R, G, B channel.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/V2/xfer_across_channels.JPG?raw=true" width="600" alt="xfer_across_channels">


### Mask Ops Node (improved for V2)
The mask ops node performs various mask operations on a mask created either from an image or a text prompt.

### Variables:
#### Separate_Mask:
The Separate_Mask option will tell the node whether to separate the mask by each island, or keep all islands in 1 image. Use 0 if you want to connect to the Color Transfer node.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/V2/separate_mask.JPG?raw=true" width="600" alt="separate_mask">

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

 ### Workflow
 A Basic workflow with all of the nodes combined has been included in the workflows directory under I2I workflow.json. Use this as a reference to see how they are all connected.

<img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/Workflow.jpg?raw=true" width="600" alt="Workflow">

 ### Color Transfer Workflow
 A Basic workflow for Color Transfer has been included in the workflows directory under Color Xfer Workflow.json. Use this as a reference to see how it works.
 
 <img src="https://github.com/ManglerFTW/ComfyI2I/blob/main/Guide_Images/ColorXferworkflow.JPG?raw=true" width="600" alt="ColorXferworkflow">
