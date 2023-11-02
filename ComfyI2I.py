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
from collections import namedtuple
import cv2
import torch
import sys
import os
import folder_paths as comfy_paths
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
import subprocess
import math

# Check for CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ARRAY_DATATYPE = torch.int32  # Corresponding to 'l'

Rgb = namedtuple('Rgb', ('r', 'g', 'b'))
Hsl = namedtuple('Hsl', ('h', 's', 'l'))

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



class Color(object):
    def __init__(self, r, g, b, proportion):
        self.rgb = Rgb(r, g, b)
        self.proportion = proportion
    
    def __repr__(self):
        return "<colorgram.py Color: {}, {}%>".format(
            str(self.rgb), str(self.proportion * 100))

    @property
    def hsl(self):
        try:
            return self._hsl
        except AttributeError:
            self._hsl = Hsl(*hsl(*self.rgb))
            return self._hsl

def extract(image_np, number_of_colors, mask_np=None):
    # Check and convert the image if needed
    if len(image_np.shape) == 2 or image_np.shape[2] != 3:  # If grayscale or not RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    
    samples = sample(image_np, mask_np)
    used = pick_used(samples)
    used.sort(key=lambda x: x[0], reverse=True)
    return get_colors(samples, used, number_of_colors)

def sample(image, mask=None):
    top_two_bits = 0b11000000

    sides = 1 << 2
    cubes = sides ** 7

    samples = torch.zeros((cubes,), dtype=torch.float32, device=device)  # Make sure samples is of float32 type

    # Handle mask
    if mask is not None:
        mask_values = (torch.rand_like(mask, dtype=torch.float32) * 255).int()
        active_pixels = mask_values > mask
    else:
        active_pixels = torch.ones_like(image[:, :, 0], dtype=torch.bool)

    # Calculate RGB, HSL, and Y
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    h, s, l = hsl(r, g, b)  # We need to convert the hsl function to use PyTorch
    Y = (r * 0.2126 + g * 0.7152 + b * 0.0722).int()

    # Packing
    packed = ((Y & top_two_bits) << 4) | ((h & top_two_bits) << 2) | (l & top_two_bits)
    packed *= 4

    # Accumulate samples
    packed_active = packed[active_pixels]
    r_active, g_active, b_active = r[active_pixels], g[active_pixels], b[active_pixels]

    samples.index_add_(0, packed_active, r_active)
    samples.index_add_(0, packed_active + 1, g_active)
    samples.index_add_(0, packed_active + 2, b_active)
    samples.index_add_(0, packed_active + 3, torch.ones_like(packed_active, dtype=torch.float32))

    return samples

def pick_used(samples):
    # Find indices where count (every 4th value) is non-zero
    non_zero_indices = torch.arange(0, samples.size(0), 4, device=samples.device)[samples[3::4] > 0]

    # Get counts for non-zero indices
    counts = samples[non_zero_indices + 3]

    # Combine counts and indices
    used = torch.stack((counts, non_zero_indices), dim=-1)

    # Convert torch tensors to list of tuples on CPU
    used_tuples = [(int(count.item()), int(idx.item())) for count, idx in zip(used[:, 0], used[:, 1])]

    return used_tuples

def get_colors(samples, used, number_of_colors):
    number_of_colors = min(number_of_colors, len(used))
    used = used[:number_of_colors]
    
    # Extract counts and indices
    counts, indices = zip(*used)
    counts = torch.tensor(counts, dtype=torch.long, device=device)
    indices = torch.tensor(indices, dtype=torch.long, device=device)

    # Calculate total pixels
    total_pixels = torch.sum(counts)

    # Get RGB values
    r_vals = samples[indices] // counts
    g_vals = samples[indices + 1] // counts
    b_vals = samples[indices + 2] // counts

    # Convert Torch tensors to lists
    r_vals_list = r_vals.tolist()
    g_vals_list = g_vals.tolist()
    b_vals_list = b_vals.tolist()
    counts_list = counts.tolist()

    # Create Color objects
    colors = [Color(r, g, b, count) for r, g, b, count in zip(r_vals_list, g_vals_list, b_vals_list, counts_list)]

    # Update proportions
    for color in colors:
        color.proportion /= total_pixels.item()

    return colors

def hsl(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    
    max_val, _ = torch.max(torch.stack([r, g, b]), dim=0)
    min_val, _ = torch.min(torch.stack([r, g, b]), dim=0)
    diff = max_val - min_val
    
    # Luminance
    l = (max_val + min_val) / 2.0

    # Saturation
    s = torch.where(
        (max_val == min_val) | (l == 0),
        torch.zeros_like(l),
        torch.where(l < 0.5, diff / (max_val + min_val), diff / (2.0 - max_val - min_val))
    )
    
    # Hue
    conditions = [
        max_val == r,
        max_val == g,
        max_val == b
    ]

    values = [
        ((g - b) / diff) % 6,
        ((b - r) / diff) + 2,
        ((r - g) / diff) + 4
    ]

    h = torch.zeros_like(r)
    for condition, value in zip(conditions, values):
        h = torch.where(condition, value, h)
    h /= 6.0

    return (h * 255).int(), (s * 255).int(), (l * 255).int()

def color_distance(pixel_color, palette_color):
    return torch.norm(pixel_color - palette_color)

def segment_image(image_torch, palette_colors, mask_torch=None, threshold=128):
    """
    Segment the image based on the color similarity of each color in the palette using PyTorch.
    """
    if mask_torch is None:
        mask_torch = torch.ones(image_torch.shape[:2], device='cuda') * 255

    output_image_torch = torch.zeros_like(image_torch)

    # Convert palette colors to PyTorch tensor
    palette_torch = torch.tensor([list(color.rgb) for color in palette_colors], device='cuda').float()

    distances = torch.norm(image_torch.unsqueeze(-2) - palette_torch, dim=-1)
    closest_color_indices = torch.argmin(distances, dim=-1)

    for idx, palette_color in enumerate(palette_torch):
        output_image_torch[closest_color_indices == idx] = palette_color

    output_image_torch[mask_torch < threshold] = image_torch[mask_torch < threshold]

    # Convert the PyTorch tensor back to a numpy array for saving or further operations
    output_image_np = output_image_torch.cpu().numpy().astype('uint8')
    return output_image_np

def calculate_luminance_vectorized(colors):
    """Calculate the luminance of an array of RGB colors using PyTorch."""
    R, G, B = colors[:, 0], colors[:, 1], colors[:, 2]
    return 0.299 * R + 0.587 * G + 0.114 * B

def luminance_match(palette1, palette2):
    # Convert palettes to PyTorch tensors
    palette1_rgb = torch.tensor([color.rgb for color in palette1], device='cuda').float()
    palette2_rgb = torch.tensor([color.rgb for color in palette2], device='cuda').float()

    luminance1 = calculate_luminance_vectorized(palette1_rgb)
    luminance2 = calculate_luminance_vectorized(palette2_rgb)

    # Sort luminances and get the sorted indices
    sorted_indices1 = torch.argsort(luminance1)
    sorted_indices2 = torch.argsort(luminance2)

    reordered_palette2 = [None] * len(palette2)

    # Match colors based on sorted luminance order
    for idx1, idx2 in zip(sorted_indices1.cpu().numpy(), sorted_indices2.cpu().numpy()):
        print(f"idx1: {idx1}, idx2: {idx2}")  # Add this to debug
        reordered_palette2[idx1] = palette2[idx2]

    return reordered_palette2

def apply_blur(image_torch, blur_radius, blur_amount):
    image_torch = image_torch.float().div(255.0)
    channels = image_torch.shape[2]

    kernel_size = int(6 * blur_radius + 1)
    kernel_size += 1 if kernel_size % 2 == 0 else 0
    
    # Calculate the padding required to keep the output size the same
    padding = kernel_size // 2
    
    # Create a Gaussian kernel
    x = torch.linspace(-blur_amount, blur_amount, kernel_size).to(image_torch.device)
    x = torch.exp(-x**2 / (2 * blur_radius**2))
    x /= x.sum()
    kernel = x[:, None] * x[None, :]
    
    # Apply the kernel using depthwise convolution
    channels = image_torch.shape[-1]
    kernel = kernel[None, None, ...].repeat(channels, 1, 1, 1)
    blurred = F.conv2d(image_torch.permute(2, 0, 1)[None, ...], kernel, groups=channels, padding=padding)
    
    # Convert the tensor back to byte and de-normalize
    blurred = (blurred * 255.0).byte().squeeze(0).permute(1, 2, 0)
    return blurred

def refined_replace_and_blend_colors(Source_np, img_np, palette1, modified_palette2, blur_radius=0, blur_amount=0, mask_torch=None):
    # Convert numpy arrays to torch tensors on GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Source = torch.from_numpy(Source_np).float().to(device)
    img_torch = torch.tensor(img_np, device=device).float()

    palette1_rgb = torch.stack([torch.tensor(color.rgb, device=device).float() if hasattr(color, 'rgb') else torch.tensor(color, device=device).float() for color in palette1])
    modified_palette2_rgb = torch.stack([torch.tensor(color.rgb, device=device).float() if hasattr(color, 'rgb') else torch.tensor(color, device=device).float() for color in modified_palette2])

    # Direct color replacement using broadcasting
    distances = torch.norm(img_torch[:, :, None] - palette1_rgb, dim=-1)
    closest_indices = torch.argmin(distances, dim=-1)
    intermediate_output = modified_palette2_rgb[closest_indices]
    
    # Convert to uint8 if not already
    intermediate_output = torch.clamp(intermediate_output, 0, 255).byte()

    # Apply blur if needed
    if blur_radius > 0 and blur_amount > 0:
        blurred_output = apply_blur(intermediate_output, blur_radius, blur_amount)
    else:
        blurred_output = intermediate_output

    # Blend based on the mask's intensity values if provided
    if mask_torch is not None:
        three_channel_mask = mask_torch[:, :, None].expand_as(Source)
        output_torch = Source * (1 - three_channel_mask) + blurred_output.float() * three_channel_mask
    else:
        output_torch = blurred_output
    
    output_np = output_torch.cpu().numpy().astype(np.uint8)

    return output_np

def torch_rgb_to_hsv(rgb):
    """
    Convert an RGB image to HSV.
    Assumes rgb is a PyTorch tensor with values in [0, 1].
    """

    # Get R, G, B values
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    max_val, _ = torch.max(rgb, dim=-1)
    min_val, _ = torch.min(rgb, dim=-1)
    diff = max_val - min_val

    # Calculate Hue
    h = torch.zeros_like(r)
    h[diff == 0] = 0
    mask = (max_val == r) & (diff != 0)
    h[mask] = (60 * ((g[mask] - b[mask]) / diff[mask]) + 360) % 360
    mask = max_val == g
    h[mask] = (60 * ((b[mask] - r[mask]) / diff[mask]) + 120) % 360
    mask = max_val == b
    h[mask] = (60 * ((r[mask] - g[mask]) / diff[mask]) + 240) % 360
    h = h / 360.  # Normalize to [0, 1]

    # Calculate Saturation
    s = torch.zeros_like(r)
    s[max_val != 0] = diff[max_val != 0] / max_val[max_val != 0]

    # Value
    v = max_val

    hsv = torch.stack([h, s, v], dim=-1)
    return hsv

def torch_hsv_to_rgb(hsv):
    """
    Convert an HSV image to RGB.
    Assumes hsv is a PyTorch tensor with values in [0, 1] for hue and [0, 1] for saturation/value.
    """

    h = hsv[..., 0] * 360.
    s = hsv[..., 1]
    v = hsv[..., 2]

    c = v * s
    hh = h / 60.
    x = c * (1 - torch.abs(hh % 2 - 1))
    m = v - c

    r, g, b = v, v, v  # Initialize with value

    mask = (hh >= 0) & (hh < 1)
    r[mask] = c[mask]
    g[mask] = x[mask]

    mask = (hh >= 1) & (hh < 2)
    r[mask] = x[mask]
    g[mask] = c[mask]

    mask = (hh >= 2) & (hh < 3)
    g[mask] = c[mask]
    b[mask] = x[mask]

    mask = (hh >= 3) & (hh < 4)
    g[mask] = x[mask]
    b[mask] = c[mask]

    mask = (hh >= 4) & (hh < 5)
    r[mask] = x[mask]
    b[mask] = c[mask]

    mask = (hh >= 5) & (hh < 6)
    r[mask] = c[mask]
    b[mask] = x[mask]

    r += m
    g += m
    b += m

    rgb = torch.stack([r, g, b], dim=-1)
    return rgb

def retain_luminance_hsv_swap(img1_np, img2_np, strength):
    """
    Blend two images while retaining the luminance of the first.
    The blending is controlled by the strength parameter.
    Assumes img1_np and img2_np are numpy arrays in BGR format.
    """

    # Convert BGR to RGB
    img1_rgb_np = cv2.cvtColor(img1_np, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    img2_rgb_np = cv2.cvtColor(img2_np, cv2.COLOR_BGR2RGB).astype(float) / 255.0

    # Blend the two RGB images linearly based on the strength
    blended_rgb_np = (1 - strength) * img1_rgb_np + strength * img2_rgb_np

    # Convert the blended RGB image and the original RGB image to YUV
    blended_yuv_np = cv2.cvtColor((blended_rgb_np * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    img1_yuv_np = cv2.cvtColor(img1_np, cv2.COLOR_BGR2YUV)

    # Replace the Y channel (luminance) of the blended image with the original image's luminance
    blended_yuv_np[:,:,0] = img1_yuv_np[:,:,0]

    # Convert back to BGR
    result_bgr_np = cv2.cvtColor(blended_yuv_np, cv2.COLOR_YUV2BGR)

    return result_bgr_np

def adjust_gamma_contrast(image_np, gamma, contrast, brightness, mask_np=None):
    # Ensure CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Transfer data to PyTorch tensors and move to the appropriate device
    image_torch = torch.tensor(image_np, dtype=torch.float32).to(device)
    
    # Gamma correction using a lookup table
    inv_gamma = 1.0 / gamma
    table = torch.tensor([(i / 255.0) ** inv_gamma * 255 for i in range(256)], device=device).float()
    gamma_corrected = torch.index_select(table, 0, image_torch.long().flatten()).reshape_as(image_torch)
    
    # Contrast and brightness adjustment
    contrast_adjusted = contrast * gamma_corrected + brightness
    contrast_adjusted = torch.clamp(contrast_adjusted, 0, 255).byte()
    
    # If mask is provided, blend the original and adjusted images
    if mask_np is not None:
        mask_torch = torch.tensor(mask_np, device=device).float() / 255.0
        three_channel_mask = mask_torch.unsqueeze(-1).expand_as(image_torch)
        contrast_adjusted = image_torch * (1 - three_channel_mask) + contrast_adjusted.float() * three_channel_mask

    # Transfer data back to numpy array
    result_np = contrast_adjusted.cpu().numpy()

    return result_np




def CutByMask(image, mask, force_resize_width, force_resize_height, mask_mapping_optional):

    if len(image.shape) < 4:
        C = 1
    else:
        C = image.shape[3]
    
    # We operate on RGBA to keep the code clean and then convert back after
    image = tensor2rgba(image)
    mask = tensor2mask(mask)

    if mask_mapping_optional is not None:
        mask_mapping_optional = mask_mapping_optional.long()
        image = image[mask_mapping_optional]

    # Scale the mask to match the image size if it isn't
    B, H, W, _ = image.shape
    mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:,0,:,:]
    
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
            resized = F.interpolate(single.permute(0, 3, 1, 2), size=(use_height, use_width), mode='bicubic').permute(0, 2, 3, 1)
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
    mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:,0,:,:]
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
                resized_image = F.interpolate(resized_image.permute(0, 3, 1, 2), size=(height,width), mode='bicubic').permute(0, 2, 3, 1)

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
                    "separate_mask": ("INT", {"default":0, "min":0, "max":1, "step":1}),
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

    RETURN_TYPES = ("IMAGE", "MASK", "MASK_MAPPING",)
    RETURN_NAMES = ("mask_image", "mask", "mask mapping")
    FUNCTION = "Mask_Ops"

    def Mask_Ops(self, image, text, separate_mask, text_sigma, use_text, blend_percentage, black_level, mid_level, white_level, channel, shrink_grow, invert=0, blur_radius=5.0, mask=None):
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

        def separate(mask, separate_flag=1):
            if separate_flag == 0:
                # Create an unseparated mapping tensor of the same length as the batch dimension
                mapping = torch.arange(mask.shape[0], device=mask.device, dtype=torch.int)
                return mask, mapping

            mask = tensor2mask(mask)
            
            thresholded = torch.gt(mask, 0).unsqueeze(1)
            B, H, W = mask.shape
            components = torch.arange(B * H * W, device=mask.device, dtype=mask.dtype).reshape(B, 1, H, W) + 1
            components[~thresholded] = 0

            while True:
                previous_components = components
                components = F.max_pool2d(components, kernel_size=3, stride=1, padding=1)
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

            return result, mapping

        image = tensor2pil(image)

        use_text = int(round(use_text))

        if use_text == 1:

            # CLIPSeg Model Loader
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
            model = "CIDAS/clipseg-rd64-refined"
            cache = os.path.join(MODELS_DIR, 'clipseg')

            inputs = CLIPSegProcessor.from_pretrained(model, cache_dir=cache)
            model = CLIPSegForImageSegmentation.from_pretrained(model, cache_dir=cache)
            
            image = image.convert('RGB')
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
                # Create a full mask for the entire image
                mask_shape = (image.size[1], image.size[0])  # Assuming image is in (H, W, C) format
                mask = torch.ones(mask_shape, dtype=torch.float32)
            else:
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


        img_b = image


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
            img = img.convert("RGB")
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

        result, mapping = separate(result, separate_mask)

        if invert == 1:
            result = 1.0 - result 

        return (result, result, mapping,)
    
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
                "no_of_colors": ("INT", {"default": 6, "min": 0, "max": 256, "step": 1}),
                "blur_radius": ("INT", {"default": 2, "min": 0, "max": 100, "step": 1}),
                "blur_amount": ("INT", {"default": 2, "min": 0, "max": 100, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 2.0, "step": 0.1}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    CATEGORY = "I2I"

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "ColorXfer2batch"

    def ColorXfer2batch(cls, source_image, target_image, no_of_colors, blur_radius, blur_amount, strength, gamma,
                        contrast,
                        brightness, mask=None):
        xfer = lambda s, t: cls.ColorXfer2(s, t, no_of_colors, blur_radius, blur_amount, strength, gamma, contrast, brightness, mask)
        type_checker = lambda img: img.unsqueeze(0) if torch.is_tensor(img) else img

        result = [xfer(type_checker(s), type_checker(t))[0] for s, t in zip(source_image, target_image)]
        return (torch.cat(result, dim=0),)

    def ColorXfer2(cls, source_image, target_image, no_of_colors, blur_radius, blur_amount, strength, gamma, contrast, brightness, mask=None):   
        if mask is not None:
            if torch.is_tensor(mask):
                # Convert to grayscale if it's a 3-channel image
                if mask.shape[-1] == 3:
                    mask = torch.mean(mask, dim=-1)

                # Remove batch dimension if present
                if mask.dim() == 3:
                    mask = mask.squeeze(0)

                mask_np1 = (mask.cpu().numpy() * 255).astype(np.uint8)
            else:
                mask_np1 = (mask * 255).astype(np.uint8)

            mask_np = mask_np1 / 255.0
            mask_torch = torch.tensor(mask_np).to(device)

        # If the source_image is a tensor, convert it to a numpy array
        if torch.is_tensor(source_image):
            Source_np = (source_image[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            Source_np = (source_image * 255).astype(np.uint8)

        # If the source_image is a tensor, convert it to a numpy array
        if torch.is_tensor(target_image):
            Target_np = (target_image[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            Target_np = (target_image * 255).astype(np.uint8)

        # Load the source image and convert to torch tensor
        Source_np = cv2.cvtColor(Source_np, cv2.COLOR_BGR2RGB)
        Source = torch.from_numpy(Source_np).float().to(device)

        # Extract colors from the source image
        colors1 = extract(Source, no_of_colors, mask_np=mask_torch)

        # Load the target image
        Target_np = cv2.cvtColor(Target_np, cv2.COLOR_BGR2RGB)
        Target = torch.from_numpy(Target_np).float().to(device=device)

        # Extract colors from the target image
        colors2 = extract(Target, no_of_colors)     

        min_length = min(len(colors1), len(colors2))
        colors1 = colors1[:min_length]
        colors2 = colors2[:min_length]

        # Segment the image
        segmented_np = segment_image(Source, colors1, mask_torch=mask_torch, threshold=1)       

        matched_pairs = luminance_match(colors1, colors2)

        result_rgb = refined_replace_and_blend_colors(Source.cpu().numpy(), segmented_np, colors1, matched_pairs, blur_radius, blur_amount, mask_torch=mask_torch)

        luminance_np = retain_luminance_hsv_swap(Source.cpu().numpy(), result_rgb, strength)

        gamma_contrast_np = adjust_gamma_contrast(luminance_np, gamma, contrast, brightness, mask_np=mask_np1)

        final_img_np_rgb = cv2.cvtColor(gamma_contrast_np, cv2.COLOR_BGR2RGB)

        # Convert the numpy array back to a PyTorch tensor
        final_img_tensor = torch.tensor(final_img_np_rgb).float().to(device)

        final_img_tensor = final_img_tensor.unsqueeze(0)

        if final_img_tensor.max() > 1.0:
            final_img_tensor /= 255.0

        return (final_img_tensor, )
 
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
