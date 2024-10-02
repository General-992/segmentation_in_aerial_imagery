import os
import zipfile
import shutil
from PIL import Image
import numpy as np

root = os.path.expanduser('~/datasets/ISPRS')
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

# images_path = os.path.join(root, 'Potsdam/img/2_Ortho_RGB')
# labels_path = os.path.join(root, 'Potsdam/msk/labels/')
#
# print(os.listdir(images_path))
#
#
def create_patches(image_name, label_name, output_img_dir, output_msk_dir, images_path, labels_path, patch_size):
    # Open image and label
    image_path = os.path.join(images_path, image_name)
    label_path = os.path.join(labels_path, label_name)
    img = Image.open(image_path)
    label = Image.open(label_path)

    # Get image dimensions
    width, height = img.size

    # Get the base name for saving patches
    if not os.path.basename(image_path).startswith('vaihingen'):
        img_base_name = os.path.basename(image_path).replace("_RGB.tif", "")
        label_base_name = os.path.basename(label_path).replace("_label.tif", "")
    else:
        img_base_name = os.path.basename(image_path).replace(".tif", "")
        label_base_name = os.path.basename(label_path).replace(".tif", "")

    # Initialize patch counter
    patch_counter = 1

    # Adjust the loop to ensure even (450x450) patches, skipping partial ones
    for i in range(0, width, patch_size):
        if i + patch_size > width:  # Skip if width exceeds
            continue
        for j in range(0, height, patch_size):
            if j + patch_size > height:  # Skip if height exceeds
                continue

            # Define the patch box (left, upper, right, lower)
            box = (i, j, i + patch_size, j + patch_size)

            # Crop the image and the label
            img_patch = img.crop(box)
            label_patch = label.crop(box)

            # Save the cropped image and label
            img_patch.save(os.path.join(output_img_dir, f"{img_base_name}_{patch_counter}_RGB.tif"))
            label_patch.save(os.path.join(output_msk_dir, f"{label_base_name}_{patch_counter}_label.tif"))

            patch_counter += 1
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
#     create_patches(image, label, output_img_dir, output_msk_dir, images_path, labels_path, patch_size=500)


img_patches = sorted(os.listdir(output_img_dir))
msk_patches = sorted(os.listdir(output_msk_dir))
output_txt_dir = os.path.join(root, 'Potsdam/test.txt')
with open(output_txt_dir, 'w') as f:
    for img, mask in zip(img_patches, msk_patches):
        assert img.replace('_RGB.tif', '') == mask.replace('_label.tif', '')
        f.write(f"{img} {mask}\n")

root = os.path.expanduser('~/datasets/ISPRS')

# zipfile_path = os.path.join(root, 'Vaihingen.zip')
#
# with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
#     zip_ref.extractall(root)
#     zip_ref.close()

root = os.path.join(root, 'Vaihingen')

# zipfile_paths = ['ISPRS_semantic_labeling_Vaihingen.zip',
#                  'ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip']
#
# for zipfile_path in zipfile_paths:
#     zipfile_path = os.path.join(root, zipfile_path)
#     with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
#         zip_ref.extractall(root)
#         zip_ref.close()

images_dir = os.path.join(root, 'img')
labels_dir = os.path.join(root, 'msk')

# os.makedirs(labels_dir, exist_ok=True)
#
# for msk in os.listdir(root):
#     if msk.startswith('top_mosaic_'):
#
#         new_msk = msk.split('aic_')[-1]
#         new_msk = f'vaihingen_msk_{new_msk}'
#
#         os.rename(os.path.join(root, msk), os.path.join(root, new_msk))
#
#         shutil.move(os.path.join(root, new_msk), labels_dir)
#         print(f'Moved {new_msk} to {labels_dir}')

os.makedirs(images_dir, exist_ok=True)

# for img in os.listdir(os.path.join(root, 'top')):
#     os.rename(os.path.join(root, f'top/{img}'), os.path.join(root, f'top/{img.replace('top_mosaic','vaihingen')}'))
#     shutil.move(os.path.join(root, f'top/{img}'), images_dir)

images_paths = sorted(os.listdir(images_dir))
labels_paths = sorted(os.listdir(labels_dir))

output_img_dir = os.path.join(root, 'img/patches')
output_msk_dir = os.path.join(root, 'msk/patches')

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_msk_dir, exist_ok=True)

# for img, label in zip(images_paths, labels_paths):
#     if img == 'patches' and label == 'patches':
#         continue
#     elif img[-5:] == label[-5:]:
#         create_patches(img, label, output_img_dir, output_msk_dir, images_dir, labels_dir, patch_size=450)

# patches_img = sorted(os.listdir(output_img_dir))
# patches_msk = sorted(os.listdir(output_msk_dir))
# output_txt_path = os.path.join(root, 'test.txt')
#
#
# with open(output_txt_path, 'w') as f:
#     for img_patch, label_patch in zip(patches_img, patches_msk):
#         if img_patch.split('_')[2] == img_patch.split('_')[2]:
#             f.write(f"{img_patch} {label_patch}\n")
#         else:
#             raise Exception(f"Image {img_patch} does not match label {label_patch}")
