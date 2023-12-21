import os
import shutil
import random

# Source folder, destination folder, and third folder
source_folder = 'd:\Combi_annotations/balanced_dataset_200'
source_copy_folder = 'd:\Combi_annotations/balanced_dataset_200_copy'
img_val_destination_folder = 'd:\Combi_annotations/train_dataset/images/val'
txt_val_destination_folder = 'd:\Combi_annotations/train_dataset/labels/val'

val_split = 0.3

# Fourth and fifth folders for the remaining 90%
img_train_folder = 'd:\Combi_annotations/train_dataset/images/train'
txt_train_folder = 'd:\Combi_annotations/train_dataset/labels/train'

# Ensure the destination and third folders exist
for folder in [img_val_destination_folder, txt_val_destination_folder, img_train_folder, txt_train_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# create a copy of input folder and operate on the copy folder 
shutil.copytree(source_folder, source_copy_folder)

source_folder = source_copy_folder


# List all files with .png extension in the source folder
png_files = [file for file in os.listdir(source_folder) if file.endswith('.png')]

# Calculate the number of files to copy (15% of total)
num_files_to_copy = int(val_split * len(png_files))

# Randomly select num_files_to_copy files to copy
files_to_copy = random.sample(png_files, num_files_to_copy)

print("Number of images Totally:", len(png_files))

# Copy the selected files and their associated text files to the destination and third folders
for file in files_to_copy:
    source_path = os.path.join(source_folder, file)
    destination_path = os.path.join(img_val_destination_folder, file)
    # third_path = os.path.join(third_folder, file)
    
    # Copy the image file to the destination folder
    shutil.move(source_path, destination_path)
    
    # Find and copy the associated text file
    txt_file = os.path.splitext(file)[0] + '.txt'
    txt_source_path = os.path.join(source_folder, txt_file)
    # txt_destination_path = os.path.join(img_val_destination_folder, txt_file)
    txt_destination_path = os.path.join(txt_val_destination_folder, txt_file)
    
    if os.path.exists(txt_source_path):
        shutil.move(txt_source_path, txt_destination_path)
        # shutil.copy(txt_source_path, txt_third_path)

    png_files.remove(file)

print("Number of images left after moving validation set:", len(png_files))
# Move the remaining 90% of images and their associated text files to the fourth and fifth folders
for file in png_files:
    source_path = os.path.join(source_folder, file)
    img_fourth_path = os.path.join(img_train_folder, file)
    
    # Move the image file to the fourth folder
    shutil.move(source_path, img_fourth_path)
    
    # Find and move the associated text file
    txt_file = os.path.splitext(file)[0] + '.txt'
    txt_source_path = os.path.join(source_folder, txt_file)
    txt_fourth_path = os.path.join(txt_train_folder, txt_file)
    
    if os.path.exists(txt_source_path):
        shutil.move(txt_source_path, txt_fourth_path)


print("moving file finished")
