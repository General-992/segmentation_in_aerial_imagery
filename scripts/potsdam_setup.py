import os
import zipfile

root = os.path.expanduser('~/datasets/ISPRS_POTSDAM')
# zipfile_path = os.path.join(root, 'Potsdam.zip')
#
# with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
#     zip_ref.extractall(root)
#
# rgb_zipfile_path = os.path.join(root, 'Potsdam/2_Ortho_RGB.zip')
#
# img_out_repo = os.path.join(root, 'Potsdam/img')
# if not os.path.exists(img_out_repo):
#     os.makedirs(img_out_repo)
#
# with zipfile.ZipFile(rgb_zipfile_path, 'r') as zip_ref:
#     zip_ref.extractall(img_out_repo)
#
# labels_zipfile_path = os.path.join(root, 'Potsdam/5_Labels_all.zip')
#
# lbl_out_repo = os.path.join(root, 'Potsdam/msk')
#
# if not os.path.exists(lbl_out_repo):
#     os.makedirs(lbl_out_repo)
#
# with zipfile.ZipFile(labels_zipfile_path, 'r') as zip_ref:
#     zip_ref.extractall(lbl_out_repo)
# from PIL import Image
# import numpy as np
# images_path = os.path.join(root, 'Potsdam/img/2_Ortho_RGB')
# labels_path = os.path.join(root, 'Potsdam/msk/labels/')
#
# print(os.listdir(images_path))
#
#
# def create_patches(image_path, label_path, output_img_dir, output_msk_dir, patch_size=500):
#     # Open image and label
#     image_path = os.path.join(images_path, image_path)
#     label_path = os.path.join(labels_path, label_path)
#     img = Image.open(image_path)
#     label = Image.open(label_path)
#
#     # Get image dimensions
#     width, height = img.size
#
#     # Get the base name for saving patches
#     img_base_name = os.path.basename(image_path).replace("_RGB.tif", "")
#     label_base_name = os.path.basename(label_path).replace("_label.tif", "")
#
#     # Initialize patch counter
#     patch_counter = 1
#
#     # Loop through the image and crop it into patches
#     for i in range(0, width, patch_size):
#         for j in range(0, height, patch_size):
#             # Define the patch box (left, upper, right, lower)
#             box = (i, j, i + patch_size, j + patch_size)
#
#             # Crop the image and the label
#             img_patch = img.crop(box)
#             label_patch = label.crop(box)
#
#             # Save the cropped image and label
#             img_patch.save(os.path.join(output_img_dir, f"{img_base_name}_{patch_counter}_RGB.tif"))
#             label_patch.save(os.path.join(output_msk_dir, f"{label_base_name}_{patch_counter}_label.tif"))
#
#             patch_counter += 1
#
output_img_dir = os.path.join(root, 'Potsdam/img/patches')
output_msk_dir = os.path.join(root, 'Potsdam/msk/patches')
#
# os.makedirs(output_img_dir, exist_ok=True)
# os.makedirs(output_msk_dir, exist_ok=True)
#
# images_paths = sorted(os.listdir(images_path))
# labels_paths = sorted(os.listdir(labels_path))
# for image, label in zip(images_paths, labels_paths):
#     create_patches(image, label, output_img_dir, output_msk_dir)

img_patches = sorted(os.listdir(output_img_dir))
msk_patches = sorted(os.listdir(output_msk_dir))
output_txt_dir = os.path.join(root, 'Potsdam/test.txt')
with open(output_txt_dir, 'w') as f:
    for img, mask in zip(img_patches, msk_patches):
        assert img.replace('_RGB.tif', '') == mask.replace('_label.tif', '')
        f.write(f"{img} {mask}\n")