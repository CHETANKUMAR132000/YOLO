import os

# Define the two folders to compare
folder1 = '/media/harsha/New Volume/Harsha_Projects/CMPV_Data/front_with_foglamp/train_data/images/val'
folder2 = '/media/harsha/New Volume/Harsha_Projects/CMPV_Data/front_with_foglamp/train_data/labels/val'

# Get a list of filenames in each folder
files1 = set([os.path.splitext(file)[0] for file in os.listdir(folder1)])
files2 = set([os.path.splitext(file)[0] for file in os.listdir(folder2)])

# Find files with mismatched names
mismatched_files = files2.symmetric_difference(files1)

# Print the mismatched files
for file in mismatched_files:
    print(f"Mismatched file name: {file}")

# If you want to include the file extensions in the comparison, remove [0] from os.path.splitext(file)[0]
