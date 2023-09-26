"""
@author: ManglerFTW
@title: ComfyI2I
"""

import shutil
import folder_paths
import os
import sys

comfy_path = os.path.dirname(folder_paths.__file__)
comfyi2i_path = os.path.join(os.path.dirname(__file__))

def setup_js():
    # remove garbage
    old_js_path = os.path.join(comfy_path, "web", "extensions", "core", "ComfyShop.js")
    if os.path.exists(old_js_path):
        os.remove(old_js_path)

    old_ip_path = os.path.join(comfy_path, "web", "extensions", "core", "imageProcessorWorker.js")
    if os.path.exists(old_ip_path):
        os.remove(old_ip_path)

    # setup js
    js_dest_path = os.path.join(comfy_path, "web", "extensions", "ComfyI2I")
    if not os.path.exists(js_dest_path):
        os.makedirs(js_dest_path)

    js_src_path = os.path.join(comfyi2i_path, "js", "ComfyShop.js")
    shutil.copy(js_src_path, js_dest_path)

    # setup ip
    ip_dest_path = os.path.join(comfy_path, "web", "extensions", "ComfyI2I")
    if not os.path.exists(ip_dest_path):
        os.makedirs(ip_dest_path)

    ip_src_path = os.path.join(comfyi2i_path, "js", "imageProcessorWorker.js")
    shutil.copy(ip_src_path, ip_dest_path)

setup_js()



from .ComfyI2I import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
