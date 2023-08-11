# By ManglerFTW (Discord: ManglerFTW)
#
# Copyright 2023 Peter Mango (ManglerFTW)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to
# deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import operator as op
import torch
import sys
import os
import folder_paths as comfy_paths
from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import to_tensor
import torchvision.transforms.functional as TF
import tensorflow as tf
import torch.nn.functional as torchfn
from PIL import Image, ImageFilter, ImageOps, ImageDraw
from torchvision.transforms import ToTensor
from torchvision import transforms
import subprocess
import math
from skimage import color, exposure

VERY_BIG_SIZE = 1024 * 1024
MAX_RESOLUTION=8192
    
MODELS_DIR =  comfy_paths.models_dir

class cstr(str):
    class color:
        END = '\33[0m'
        BOLD = '\33[1m'
        ITALIC = '\33[3m'
        UNDERLINE = '\33[4m'
        BLINK = '\33[5m'
        BLINK2 = '\33[6m'
        SELECTED = '\33[7m'

        BLACK = '\33[30m'
        RED = '\33[31m'
        GREEN = '\33[32m'
        YELLOW = '\33[33m'
        BLUE = '\33[34m'
        VIOLET = '\33[35m'
        BEIGE = '\33[36m'
        WHITE = '\33[37m'

        BLACKBG = '\33[40m'
        REDBG = '\33[41m'
        GREENBG = '\33[42m'
        YELLOWBG = '\33[43m'
        BLUEBG = '\33[44m'
        VIOLETBG = '\33[45m'
        BEIGEBG = '\33[46m'
        WHITEBG = '\33[47m'

        GREY = '\33[90m'
        LIGHTRED = '\33[91m'
        LIGHTGREEN = '\33[92m'
        LIGHTYELLOW = '\33[93m'
        LIGHTBLUE = '\33[94m'
        LIGHTVIOLET = '\33[95m'
        LIGHTBEIGE = '\33[96m'
        LIGHTWHITE = '\33[97m'

        GREYBG = '\33[100m'
        LIGHTREDBG = '\33[101m'
        LIGHTGREENBG = '\33[102m'
        LIGHTYELLOWBG = '\33[103m'
        LIGHTBLUEBG = '\33[104m'
        LIGHTVIOLETBG = '\33[105m'
        LIGHTBEIGEBG = '\33[106m'
        LIGHTWHITEBG = '\33[107m'

        @staticmethod
        def add_code(name, code):
            if not hasattr(cstr.color, name.upper()):
                setattr(cstr.color, name.upper(), code)
            else:
                raise ValueError(f"'cstr' object already contains a code with the name '{name}'.")

    def __new__(cls, text):
        return super().__new__(cls, text)

    def __getattr__(self, attr):
        if attr.lower().startswith("_cstr"):
            code = getattr(self.color, attr.upper().lstrip("_cstr"))
            modified_text = self.replace(f"__{attr[1:]}__", f"{code}")
            return cstr(modified_text)
        elif attr.upper() in dir(self.color):
            code = getattr(self.color, attr.upper())
            modified_text = f"{code}{self}{self.color.END}"
            return cstr(modified_text)
        elif attr.lower() in dir(cstr):
            return getattr(cstr, attr.lower())
        else:
            raise AttributeError(f"'cstr' object has no attribute '{attr}'")

    def print(self, **kwargs):
        print(self, **kwargs)

def tensor2rgb(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 3)
    if size[3] == 1:
        return t.repeat(1, 1, 1, 3)
    elif size[3] == 4:
        return t[:, :, :, :3]
    else:
        return t
    
def tensor2rgba(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 4)
    elif size[3] == 1:
        return t.repeat(1, 1, 1, 4)
    elif size[3] == 3:
        alpha_tensor = torch.ones((size[0], size[1], size[2], 1))
        return torch.cat((t, alpha_tensor), dim=3)
    else:
        return t

def tensor2mask(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t
    if size[3] == 1:
        return t[:,:,:,0]
    elif size[3] == 4:
        # Not sure what the right thing to do here is. Going to try to be a little smart and use alpha unless all alpha is 1 in case we'll fallback to RGB behavior
        if torch.min(t[:, :, :, 3]).item() != 1.:
            return t[:,:,:,3]

    return TF.rgb_to_grayscale(tensor2rgb(t).permute(0,3,1,2), num_output_channels=1)[:,0,:,:]

def tensor2batch(t: torch.Tensor, bs: torch.Size) -> torch.Tensor:
    if len(t.size()) < len(bs):
        t = t.unsqueeze(3)
    if t.size()[0] < bs[0]:
        t.repeat(bs[0], 1, 1, 1)
    dim = bs[3]
    if dim == 1:
        return tensor2mask(t)
    elif dim == 3:
        return tensor2rgb(t)
    elif dim == 4:
        return tensor2rgba(t)

def tensors2common(t1: torch.Tensor, t2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    t1s = t1.size()
    t2s = t2.size()
    if len(t1s) < len(t2s):
        t1 = t1.unsqueeze(3)
    elif len(t1s) > len(t2s):
        t2 = t2.unsqueeze(3)

    if len(t1.size()) == 3:
        if t1s[0] < t2s[0]:
            t1 = t1.repeat(t2s[0], 1, 1)
        elif t1s[0] > t2s[0]:
            t2 = t2.repeat(t1s[0], 1, 1)
    else:
        if t1s[0] < t2s[0]:
            t1 = t1.repeat(t2s[0], 1, 1, 1)
        elif t1s[0] > t2s[0]:
            t2 = t2.repeat(t1s[0], 1, 1, 1)

    t1s = t1.size()
    t2s = t2.size()
    if len(t1s) > 3 and t1s[3] < t2s[3]:
        return tensor2batch(t1, t2s), t2
    elif len(t1s) > 3 and t1s[3] > t2s[3]:
        return t1, tensor2batch(t2, t1s)
    else:
        return t1, t2

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# PIL to Tensor
def pil2tensor_stacked(image):
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    elif isinstance(image, torch.Tensor):
        return image
    else:
        raise ValueError(f"Unexpected datatype for input to 'pil2tensor_stacked'. Expected a PIL Image or tensor, but received type: {type(image)}")

def CutByMask(image, mask, force_resize_width, force_resize_height, mask_mapping_optional):

    if len(image.shape) < 4:
        C = 1
    else:
        C = image.shape[3]
    
    # We operate on RGBA to keep the code clean and then convert back after
    image = tensor2rgba(image)
    mask = tensor2mask(mask)

    if mask_mapping_optional is not None:
        image = image[mask_mapping_optional]

    # Scale the mask to match the image size if it isn't
    B, H, W, _ = image.shape
    mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:,0,:,:]
    
    MB, _, _ = mask.shape

    if MB < B:
        assert(B % MB == 0)
        mask = mask.repeat(B // MB, 1, 1)

    # Masks to boxes
    is_empty = ~torch.gt(torch.max(torch.reshape(mask, [B, H * W]), dim=1).values, 0.)
    mask[is_empty,0,0] = 1.
    boxes = masks_to_boxes(mask)
    mask[is_empty,0,0] = 0.

    min_x = boxes[:,0]
    min_y = boxes[:,1]
    max_x = boxes[:,2]
    max_y = boxes[:,3]

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    use_width = int(torch.max(width).item())
    use_height = int(torch.max(height).item())

    if force_resize_width > 0:
        use_width = force_resize_width

    if force_resize_height > 0:
        use_height = force_resize_height

    print("use_width: ", use_width)
    print("use_height: ", use_height)

    alpha_mask = torch.ones((B, H, W, 4))
    alpha_mask[:,:,:,3] = mask

    image = image * alpha_mask

    result = torch.zeros((B, use_height, use_width, 4))
    for i in range(0, B):
        if not is_empty[i]:
            ymin = int(min_y[i].item())
            ymax = int(max_y[i].item())
            xmin = int(min_x[i].item())
            xmax = int(max_x[i].item())
            single = (image[i, ymin:ymax+1, xmin:xmax+1,:]).unsqueeze(0)
            resized = torch.nn.functional.interpolate(single.permute(0, 3, 1, 2), size=(use_height, use_width), mode='bicubic').permute(0, 2, 3, 1)
            result[i] = resized[0]

    # Preserve our type unless we were previously RGB and added non-opaque alpha due to the mask size
    if C == 1:
        print("C == 1 output image shape: ", tensor2mask(result).shape)
        return tensor2mask(result)
    elif C == 3 and torch.min(result[:,:,:,3]) == 1:
        print("C == 3 output image shape: ", tensor2rgb(result).shape)
        return tensor2rgb(result)
    else:
        print("else result shape: ", result.shape)
        return result

def combine(image1, image2, op, clamp_result, round_result):
    image1, image2 = tensors2common(image1, image2)

    if op == "union (max)":
        result = torch.max(image1, image2)
    elif op == "intersection (min)":
        result = torch.min(image1, image2)
    elif op == "difference":
        result = image1 - image2
    elif op == "multiply":
        result = image1 * image2
    elif op == "multiply_alpha":
        image1 = tensor2rgba(image1)
        image2 = tensor2mask(image2)
        result = torch.cat((image1[:, :, :, :3], (image1[:, :, :, 3] * image2).unsqueeze(3)), dim=3)
    elif op == "add":
        result = image1 + image2
    elif op == "greater_or_equal":
        result = torch.where(image1 >= image2, 1., 0.)
    elif op == "greater":
        result = torch.where(image1 > image2, 1., 0.)

    if clamp_result == "yes":
        result = torch.min(torch.max(result, torch.tensor(0.)), torch.tensor(1.))
    if round_result == "yes":
        result = torch.round(result)

    return result

def apply_color_correction(target_image, source_image, factor=1):

    if not isinstance(source_image, (torch.Tensor, Image.Image)):
        raise ValueError(f"Unexpected datatype for 'source_image' at method start. Expected a tensor or PIL Image, but received type: {type(source_image)}")
    
    # Ensure source_image is a tensor
    if isinstance(source_image, Image.Image):  # Check if it's a PIL Image
        source_image = pil2tensor_stacked(source_image)  # Convert it to tensor

    if not isinstance(source_image, (torch.Tensor, Image.Image)):
        raise ValueError(f"Unexpected datatype for 'source_image'. Expected a tensor or PIL Image, but received type: {type(source_image)}")

    # Get the batch size
    batch_size = source_image.shape[0]
    output_images = []

    for i in range(batch_size):
        # Convert the source and target images to NumPy arrays for the i-th image in the batch
        source_numpy = source_image[i, ...].numpy()
        target_numpy = target_image[i, ...].numpy()

        # Convert to float32
        source_numpy = source_numpy.astype(np.float32)
        target_numpy = target_numpy.astype(np.float32)

        # If the images have an alpha channel, remove it for the color transformations
        if source_numpy.shape[-1] == 4:
            source_numpy = source_numpy[..., :3]
        if target_numpy.shape[-1] == 4:
            target_numpy = target_numpy[..., :3]

        # Compute the mean and standard deviation of the color channels for both images
        target_mean, target_std = np.mean(source_numpy, axis=(0, 1)), np.std(source_numpy, axis=(0, 1))
        source_mean, source_std = np.mean(target_numpy, axis=(0, 1)), np.std(target_numpy, axis=(0, 1))

        adjusted_source_mean = target_mean + factor * (target_mean - source_mean)
        adjusted_source_std = target_std + factor * (target_std - source_std)

        # Normalize the target image (zero mean and unit variance)
        target_norm = (target_numpy - target_mean) / target_std

        # Scale and shift the normalized target image to match the exaggerated source image statistics
        matched_rgb = target_norm * adjusted_source_std + adjusted_source_mean

        # Clip values to [0, 1] and convert to PIL Image
        img = Image.fromarray(np.clip(matched_rgb * 255, 0, 255).astype('uint8'), 'RGB')

        # Convert the PIL Image to a tensor and append to the list
        img_tensor = pil2tensor_stacked(img)
        output_images.append(img_tensor)

    # Stack the list of tensors to get the batch of corrected images
    stacked_images = torch.stack(output_images)

    return stacked_images

def PasteByMask(image_base, image_to_paste, mask, resize_behavior, mask_mapping_optional):
    image_base = tensor2rgba(image_base)
    image_to_paste = tensor2rgba(image_to_paste)
    mask = tensor2mask(mask)

    # Scale the mask to be a matching size if it isn't
    B, H, W, C = image_base.shape
    MB = mask.shape[0]
    PB = image_to_paste.shape[0]
    if mask_mapping_optional is None:
        if B < PB:
            assert(PB % B == 0)
            image_base = image_base.repeat(PB // B, 1, 1, 1)
        B, H, W, C = image_base.shape
        if MB < B:
            assert(B % MB == 0)
            mask = mask.repeat(B // MB, 1, 1)
        elif B < MB:
            assert(MB % B == 0)
            image_base = image_base.repeat(MB // B, 1, 1, 1)
        if PB < B:
            assert(B % PB == 0)
            image_to_paste = image_to_paste.repeat(B // PB, 1, 1, 1)
    mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:,0,:,:]
    MB, MH, MW = mask.shape

    # masks_to_boxes errors if the tensor is all zeros, so we'll add a single pixel and zero it out at the end
    is_empty = ~torch.gt(torch.max(torch.reshape(mask,[MB, MH * MW]), dim=1).values, 0.)
    mask[is_empty,0,0] = 1.
    boxes = masks_to_boxes(mask)
    mask[is_empty,0,0] = 0.

    min_x = boxes[:,0]
    min_y = boxes[:,1]
    max_x = boxes[:,2]
    max_y = boxes[:,3]
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    target_width = max_x - min_x + 1
    target_height = max_y - min_y + 1

    result = image_base.detach().clone()

    for i in range(0, MB):
        if i >= len(image_to_paste):
            raise ValueError(f"image_to_paste does not have an entry for mask index {i}")
        if is_empty[i]:
            continue
        else:
            image_index = i
            if mask_mapping_optional is not None:
                image_index = mask_mapping_optional[i].item()
            source_size = image_to_paste.size()
            SB, SH, SW, _ = image_to_paste.shape

            # Figure out the desired size
            width = int(target_width[i].item())
            height = int(target_height[i].item())
            if resize_behavior == "keep_ratio_fill":
                target_ratio = width / height
                actual_ratio = SW / SH
                if actual_ratio > target_ratio:
                    width = int(height * actual_ratio)
                elif actual_ratio < target_ratio:
                    height = int(width / actual_ratio)
            elif resize_behavior == "keep_ratio_fit":
                target_ratio = width / height
                actual_ratio = SW / SH
                if actual_ratio > target_ratio:
                    height = int(width / actual_ratio)
                elif actual_ratio < target_ratio:
                    width = int(height * actual_ratio)
            elif resize_behavior == "source_size" or resize_behavior == "source_size_unmasked":
                width = SW
                height = SH

            # Resize the image we're pasting if needed
            resized_image = image_to_paste[i].unsqueeze(0)
            if SH != height or SW != width:
                resized_image = torch.nn.functional.interpolate(resized_image.permute(0, 3, 1, 2), size=(height,width), mode='bicubic').permute(0, 2, 3, 1)

            pasting = torch.ones([H, W, C])
            ymid = float(mid_y[i].item())
            ymin = int(math.floor(ymid - height / 2)) + 1
            ymax = int(math.floor(ymid + height / 2)) + 1
            xmid = float(mid_x[i].item())
            xmin = int(math.floor(xmid - width / 2)) + 1
            xmax = int(math.floor(xmid + width / 2)) + 1

            _, source_ymax, source_xmax, _ = resized_image.shape
            source_ymin, source_xmin = 0, 0

            if xmin < 0:
                source_xmin = abs(xmin)
                xmin = 0
            if ymin < 0:
                source_ymin = abs(ymin)
                ymin = 0
            if xmax > W:
                source_xmax -= (xmax - W)
                xmax = W
            if ymax > H:
                source_ymax -= (ymax - H)
                ymax = H

            pasting[ymin:ymax, xmin:xmax, :] = resized_image[0, source_ymin:source_ymax, source_xmin:source_xmax, :]
            pasting[:, :, 3] = 1.

            pasting_alpha = torch.zeros([H, W])
            pasting_alpha[ymin:ymax, xmin:xmax] = resized_image[0, source_ymin:source_ymax, source_xmin:source_xmax, 3]

            if resize_behavior == "keep_ratio_fill" or resize_behavior == "source_size_unmasked":
                # If we explicitly want to fill the area, we are ok with extending outside
                paste_mask = pasting_alpha.unsqueeze(2).repeat(1, 1, 4)
            else:
                paste_mask = torch.min(pasting_alpha, mask[i]).unsqueeze(2).repeat(1, 1, 4)
            result[image_index] = pasting * paste_mask + result[image_index] * (1. - paste_mask)
    return result

class Mask_Ops:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
                "required": {
                    "image": ("IMAGE",),
                    "text": ("STRING", {"default":"", "multiline": False}),
                    "text_sigma": ("INT", {"default":30, "min":0, "max":150, "step":1}),
                    "use_text": ("INT", {"default":0, "min":0, "max":1, "step":1}),
                    "blend_percentage": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "black_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255.0, "step": 0.1}),
                    "mid_level": ("FLOAT", {"default": 127.5, "min": 0.0, "max": 255.0, "step": 0.1}),
                    "white_level": ("FLOAT", {"default": 255, "min": 0.0, "max": 255.0, "step": 0.1}),
                    "channel": (["red", "green", "blue"],),
                    "shrink_grow": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                    "invert": ("INT", {"default":0, "min":0, "max":1, "step":1}),
                    "blur_radius": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1024, "step": 0.1}),
                },
                "optional": {
                    "mask": ("MASK",),
                },
        }

    CATEGORY = "I2I"

    RETURN_TYPES = ("IMAGE","MASK_MAPPING",)
    RETURN_NAMES = ("mask", "mask mapping")
    FUNCTION = "Mask_Ops"

    def Mask_Ops(self, image, text, text_sigma, use_text, blend_percentage, black_level, mid_level, white_level, channel, shrink_grow, invert=0, blur_radius=5.0, mask=None):
        channels = ["red", "green", "blue"]

        # Freeze PIP modules
        def packages(versions=False):
            import sys
            import subprocess
            return [( r.decode().split('==')[0] if not versions else r.decode() ) for r in subprocess.check_output([sys.executable, '-s', '-m', 'pip', 'freeze']).split()]

        # PIL to Mask
        def pil2mask(image):
            image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
            mask = torch.from_numpy(image_np)
            return 1.0 - mask

        def gaussian_region(image, radius=5.0):
            image = ImageOps.invert(image.convert("L"))
            image = image.filter(ImageFilter.GaussianBlur(radius=int(radius)))
            return image.convert("RGB")

        # scipy handling
        if 'scipy' not in packages():
            cstr("Installing `scipy` ...").msg.print()
            subprocess.check_call([sys.executable, '-s', '-m', 'pip', 'install', 'scipy'])
            try:
                import scipy
            except ImportError as e:
                cstr("Unable to import tools for certain masking procedures.").msg.print()
                print(e)

        def smooth_region(image, tolerance):
            from scipy.ndimage import gaussian_filter
            image = image.convert("L")
            mask_array = np.array(image)
            smoothed_array = gaussian_filter(mask_array, sigma=tolerance)
            threshold = np.max(smoothed_array) / 2
            smoothed_mask = np.where(smoothed_array >= threshold, 255, 0).astype(np.uint8)
            smoothed_image = Image.fromarray(smoothed_mask, mode="L")
            return ImageOps.invert(smoothed_image.convert("RGB"))

        def erode_region(image, iterations):
            from scipy.ndimage import binary_erosion
            image = image.convert("L")
            binary_mask = np.array(image) > 0
            eroded_mask = binary_erosion(binary_mask, iterations=iterations)
            eroded_image = Image.fromarray(eroded_mask.astype(np.uint8) * 255, mode="L")
            return ImageOps.invert(eroded_image.convert("RGB"))

        def dilate_region(image, iterations):
            from scipy.ndimage import binary_dilation
            image = image.convert("L")
            binary_mask = np.array(image) > 0
            dilated_mask = binary_dilation(binary_mask, iterations=iterations)
            dilated_image = Image.fromarray(dilated_mask.astype(np.uint8) * 255, mode="L")
            return ImageOps.invert(dilated_image.convert("RGB"))

        def erode(masks, iterations):
            iterations = iterations * -1
            if masks.ndim > 3:
                regions = []
                for mask in masks:
                    mask_np = np.clip(255. * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                    pil_image = Image.fromarray(mask_np, mode="L")
                    region_mask = erode_region(pil_image, iterations)
                    region_tensor = pil2mask(region_mask).unsqueeze(0).unsqueeze(1)
                    regions.append(region_tensor)
                regions_tensor = torch.cat(regions, dim=0)
                return regions_tensor
            else:
                mask_np = np.clip(255. * masks.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(mask_np, mode="L")
                region_mask = erode_region(pil_image, iterations)
                region_tensor = pil2mask(region_mask).unsqueeze(0).unsqueeze(1)
                return region_tensor

        def dilate(masks, iterations):
            if masks.ndim > 3:
                regions = []
                for mask in masks:
                    mask_np = np.clip(255. * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                    pil_image = Image.fromarray(mask_np, mode="L")
                    region_mask = dilate_region(pil_image, iterations)
                    region_tensor = pil2mask(region_mask).unsqueeze(0).unsqueeze(1)
                    regions.append(region_tensor)
                regions_tensor = torch.cat(regions, dim=0)
                return regions_tensor
            else:
                mask_np = np.clip(255. * masks.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(mask_np, mode="L")
                region_mask = dilate_region(pil_image, iterations)
                region_tensor = pil2mask(region_mask).unsqueeze(0).unsqueeze(1)
                return region_tensor

        def separate(mask):
            mask = tensor2mask(mask)

            
            thresholded = torch.gt(mask,0).unsqueeze(1)
            B, H, W = mask.shape
            components = torch.arange(B * H * W, device=mask.device, dtype=mask.dtype).reshape(B, 1, H, W) + 1
            components[~thresholded] = 0

            while True:
                previous_components = components
                components = torch.nn.functional.max_pool2d(components, kernel_size=3, stride=1, padding=1)
                components[~thresholded] = 0
                if torch.equal(previous_components, components):
                    break

            components = components.reshape(B, H, W)
            segments = torch.unique(components)
            result = torch.zeros([len(segments) - 1, H, W])
            index = 0
            mapping = torch.zeros([len(segments) - 1], device=mask.device, dtype=torch.int)
            for i in range(len(segments)):
                segment = segments[i].item()
                if segment == 0:
                    continue
                image_index = int((segment - 1) // (H * W))
                segment_mask = (components[image_index,:,:] == segment)
                result[index][segment_mask] = mask[image_index][segment_mask]
                mapping[index] = image_index
                index += 1

            return result,mapping

        use_text = int(round(use_text))

        if use_text == 1:

            # CLIPSeg Model Loader
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
            model = "CIDAS/clipseg-rd64-refined"
            cache = os.path.join(MODELS_DIR, 'clipseg')

            inputs = CLIPSegProcessor.from_pretrained(model, cache_dir=cache)
            model = CLIPSegForImageSegmentation.from_pretrained(model, cache_dir=cache)
            
            image = tensor2pil(image)
            cache = os.path.join(MODELS_DIR, 'clipseg')

            with torch.no_grad():
                result = model(**inputs(text=text, images=image, padding=True, return_tensors="pt"))

            tensor = torch.sigmoid(result[0])
            mask = 1. - (tensor - tensor.min()) / tensor.max()
            mask = mask.unsqueeze(0)
            mask = tensor2pil(mask).convert("L")
            mask = mask.resize(image.size)
                    
            sigma = text_sigma
            mask = pil2mask(mask)   

            if mask.ndim > 3:
                regions = []
                for mk in mask:
                    mask_np = np.clip(255. * mk.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                    pil_image = Image.fromarray(mask_np, mode="L")
                    region_mask = smooth_region(pil_image, sigma)
                    region_tensor = pil2mask(region_mask).unsqueeze(0).unsqueeze(1)
                    regions.append(region_tensor)
                mask = torch.cat(regions, dim=0)
   
            else:
                mask_np = np.clip(255. * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(mask_np, mode="L")
                region_mask = smooth_region(pil_image, sigma)
                mask = pil2mask(region_mask).unsqueeze(0).unsqueeze(1)

        else:
            if mask is None:
                raise ValueError("A mask is required when use_text is 0.")
            
            mask = mask

        if shrink_grow < 0:
            mask = erode(mask, shrink_grow)
        elif shrink_grow > 0:
            mask = dilate(mask, shrink_grow)

        invert = int(round(invert))
        if invert == 1:
            mask = 1.0 - mask

        #Invert Mask
        Mask_Inv = mask
        #Convert Inverted Mask to Image
        Inv_Mask_2_Img = Mask_Inv.reshape((-1, 1, Mask_Inv.shape[-2], Mask_Inv.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        
        #Convert Mask to Image
        Mask_2_Img = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        
        #Image Blend by Mask
        # Convert images to PIL
        img_a = tensor2pil(Inv_Mask_2_Img)

        if use_text == 1:
            img_b = image
        else:
            img_b = tensor2pil(image)

        mask = ImageOps.invert(tensor2pil(Mask_2_Img).convert('L'))

        # Mask image
        masked_img = Image.composite(img_a, img_b, mask.resize(img_a.size))

        # Blend image
        blend_mask = Image.new(mode="L", size=img_a.size,
                               color=(round(blend_percentage * 255)))
        blend_mask = ImageOps.invert(blend_mask)
        Blended_Image = Image.composite(img_a, masked_img, blend_mask)

        Blended_Image = pil2tensor(Blended_Image)

        del img_a, img_b, blend_mask, mask

        #Image Levels Adjustment
        # Convert image to PIL
        tensor_images = []
        for img in Blended_Image:
            img = tensor2pil(img)
            levels = self.AdjustLevels(black_level, mid_level, white_level)
            tensor_images.append(pil2tensor(levels.adjust(img)))
        tensor_images = torch.cat(tensor_images, dim=0)

        #Convert Image to Mask
        masks = tensor_images[0, :, :, channels.index(channel)]

        if masks.ndim > 3:
            regions = []
            for mask in masks:
                mask_np = np.clip(255. * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(mask_np, mode="L")
                region_mask = gaussian_region(pil_image, blur_radius)
                region_tensor = pil2mask(region_mask).unsqueeze(0).unsqueeze(1)
                regions.append(region_tensor)
            result = torch.cat(regions, dim=0)

        else:
            mask_np = np.clip(255. * masks.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(mask_np, mode="L")
            region_mask = gaussian_region(pil_image, blur_radius)
            result = pil2mask(region_mask).unsqueeze(0).unsqueeze(1)

        result = result.reshape((-1, 1, result.shape[-2], result.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

        if invert == 1:
            result = 1.0 - result       

        result, mapping = separate(result)

        if invert == 1:
            result = 1.0 - result 

        return (result,mapping,)
    
    class AdjustLevels:
        def __init__(self, min_level, mid_level, max_level):
            self.min_level = min_level
            self.mid_level = mid_level
            self.max_level = max_level

        def adjust(self, im):

            im_arr = np.array(im)
            im_arr[im_arr < self.min_level] = self.min_level
            im_arr = (im_arr - self.min_level) * \
                (255 / (self.max_level - self.min_level))
            im_arr[im_arr < 0] = 0
            im_arr[im_arr > 255] = 255
            im_arr = im_arr.astype(np.uint8)
            
            im = Image.fromarray(im_arr)
            im = ImageOps.autocontrast(im, cutoff=self.max_level)

            return im
        
        
class Color_Correction:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
        }

    CATEGORY = "I2I"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image")
    FUNCTION = "ColorXfer"

    def ColorXfer(cls, target_image, source_image, factor=1):
        result = apply_color_correction(target_image, source_image, factor)
        return (result,)  
    
class MaskToRegion:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "force_resize_width": ("INT", {"default": 1024, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "force_resize_height": ("INT", {"default": 1024, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "kind": (["mask", "RGB", "RGBA"],),
                "padding": ("INT", {"default": 3, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "constraints": (["keep_ratio", "keep_ratio_divisible", "multiple_of", "ignore"],),
                "constraint_x": ("INT", {"default": 64, "min": 2, "max": VERY_BIG_SIZE, "step": 1}),
                "constraint_y": ("INT", {"default": 64, "min": 2, "max": VERY_BIG_SIZE, "step": 1}),
                "min_width": ("INT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "min_height": ("INT", {"default": 0, "min": 0, "max": VERY_BIG_SIZE, "step": 1}),
                "batch_behavior": (["match_ratio", "match_size"],),
            },
            "optional": {
                "mask_mapping_optional": ("MASK_MAPPING",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", )
    RETURN_NAMES = ("cut image", "cut mask", "region")
    FUNCTION = "get_region"

    CATEGORY = "I2I"

    def get_region(self, image, mask, force_resize_width, force_resize_height, kind, padding, constraints, constraint_x, constraint_y, min_width, min_height, batch_behavior, mask_mapping_optional = None):
        mask2 = tensor2mask(mask)
        mask_size = mask2.size()
        mask_width = int(mask_size[2])
        mask_height = int(mask_size[1])

        # masks_to_boxes errors if the tensor is all zeros, so we'll add a single pixel and zero it out at the end        
        is_empty = ~torch.gt(torch.max(torch.reshape(mask2,[mask_size[0], mask_width * mask_height]), dim=1).values, 0.)
        mask2[is_empty,0,0] = 1.
        boxes = masks_to_boxes(mask2)
        mask2[is_empty,0,0] = 0.

        # Account for padding
        min_x = torch.max(boxes[:,0] - padding, torch.tensor(0.))
        min_y = torch.max(boxes[:,1] - padding, torch.tensor(0.))
        max_x = torch.min(boxes[:,2] + padding, torch.tensor(mask_width))
        max_y = torch.min(boxes[:,3] + padding, torch.tensor(mask_height))

        width = max_x - min_x
        height = max_y - min_y

        # Make sure the width and height are big enough
        target_width = torch.max(width, torch.tensor(min_width))
        target_height = torch.max(height, torch.tensor(min_height))

        if constraints == "keep_ratio":
            target_width = torch.max(target_width, target_height * constraint_x // constraint_y)
            target_height = torch.max(target_height, target_width * constraint_y // constraint_x)
        elif constraints == "keep_ratio_divisible":
            # Probably a more efficient way to do this, but given the bounds it's not too bad
            max_factors = torch.min(constraint_x // target_width, constraint_y // target_height)
            max_factor = int(torch.max(max_factors).item())
            for i in range(1, max_factor+1):
                divisible = constraint_x % i == 0 and constraint_y % i == 0
                if divisible:
                    big_enough = ~torch.lt(target_width, constraint_x // i) * ~torch.lt(target_height, constraint_y // i)
                    target_width[big_enough] = constraint_x // i
                    target_height[big_enough] = constraint_y // i
        elif constraints == "multiple_of":
            target_width[torch.gt(target_width % constraint_x, 0)] = (target_width // constraint_x + 1) * constraint_x
            target_height[torch.gt(target_height % constraint_y, 0)] = (target_height // constraint_y + 1) * constraint_y

        if batch_behavior == "match_size":
            target_width[:] = torch.max(target_width)
            target_height[:] = torch.max(target_height)
        elif batch_behavior == "match_ratio":
            # We'll target the ratio that's closest to 1:1, but don't want to take into account empty masks
            ratios = torch.abs(target_width / target_height - 1)
            ratios[is_empty] = 10000
            match_ratio = torch.min(ratios,dim=0).indices.item()
            target_width = torch.max(target_width, target_height * target_width[match_ratio] // target_height[match_ratio])
            target_height = torch.max(target_height, target_width * target_height[match_ratio] // target_width[match_ratio])

        missing = target_width - width
        min_x = min_x - missing // 2
        max_x = max_x + (missing - missing // 2)

        missing = target_height - height
        min_y = min_y - missing // 2
        max_y = max_y + (missing - missing // 2)

        # Move the region into range if needed
        bad = torch.lt(min_x,0)
        max_x[bad] -= min_x[bad]
        min_x[bad] = 0

        bad = torch.lt(min_y,0)
        max_y[bad] -= min_y[bad]
        min_y[bad] = 0

        bad = torch.gt(max_x, mask_width)
        min_x[bad] -= (max_x[bad] - mask_width)
        max_x[bad] = mask_width

        bad = torch.gt(max_y, mask_height)
        min_y[bad] -= (max_y[bad] - mask_height)
        max_y[bad] = mask_height

        region = torch.zeros((mask_size[0], mask_height, mask_width))
        for i in range(0, mask_size[0]):
            if not is_empty[i]:
                ymin = int(min_y[i].item())
                ymax = int(max_y[i].item())
                xmin = int(min_x[i].item())
                xmax = int(max_x[i].item())
                region[i, ymin:ymax+1, xmin:xmax+1] = 1

        Cut_Image = CutByMask(image, region, force_resize_width, force_resize_height, mask_mapping_optional)
        
        #Change Channels >>>> OUTPUT TO VAE ENCODE
        if kind == "mask":
            Cut_Image = tensor2mask(Cut_Image)
        elif kind == "RGBA":
            Cut_Image = tensor2rgba(Cut_Image)
        else: # RGB
            Cut_Image = tensor2rgb(Cut_Image)

        Cut_Mask = CutByMask(mask, region, force_resize_width, force_resize_height, mask_mapping_optional = None)

        return (Cut_Image, Cut_Mask, region, )

class Combine_And_Paste_Op:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoded_vae": ("IMAGE",),
                "Original_Image": ("IMAGE",),
                "Cut_Image": ("IMAGE",),
                "Cut_Mask": ("IMAGE",),
                "region": ("IMAGE",),
                "color_xfer_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "op": (["union (max)", "intersection (min)", "difference", "multiply", "multiply_alpha", "add", "greater_or_equal", "greater"],),
                "clamp_result": (["yes", "no"],),
                "round_result": (["no", "yes"],),
                "resize_behavior": (["resize", "keep_ratio_fill", "keep_ratio_fit", "source_size", "source_size_unmasked"],),
            },
            "optional": {
                "mask_mapping_optional": ("MASK_MAPPING",),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("FinalOut", )
    FUNCTION = "com_paste_op"

    CATEGORY = "I2I"

    def com_paste_op(self, decoded_vae, Original_Image, Cut_Image, Cut_Mask, region, color_xfer_factor, op, clamp_result, round_result, resize_behavior, mask_mapping_optional = None):

        Combined_Decoded = combine(decoded_vae, Cut_Mask, op, clamp_result, round_result)

        Combined_Originals = combine(Cut_Image, Cut_Mask, op, clamp_result, round_result)

        Cx_Decoded = apply_color_correction(Combined_Decoded, Combined_Originals, color_xfer_factor)

        Cx_Decode_Mask = combine(Cx_Decoded, Cut_Mask, op, clamp_result, round_result)

        FinalOut = PasteByMask(Original_Image, Cx_Decode_Mask, region, resize_behavior, mask_mapping_optional)

        return (FinalOut, )

NODE_CLASS_MAPPINGS = {
    "Color Transfer": Color_Correction,
    "Mask Ops": Mask_Ops,
    "Inpaint Segments": MaskToRegion,
    "Combine and Paste": Combine_And_Paste_Op,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Color Transfer": "Color Transfer",
    "Mask Ops": "Mask Ops",
    "Inpaint Segments": "Inpaint Segments",
    "Combine and Paste": "Combine and Paste",
}
