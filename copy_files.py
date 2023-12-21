import os
import shutil

def copy_files(src_dir, dest_dir):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Walk through the source directory and its subdirectories
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # Construct the full path of the source file
            src_file = os.path.join(root, file)

            # Construct the corresponding path in the destination directory
            dest_file = os.path.join(dest_dir, file)

            # Copy the file
            shutil.copy2(src_file, dest_file)
            print(f"Copied: {src_file} to {dest_file}")

if __name__ == "__main__":
    # Replace these paths with your actual source and destination paths
    source_directory = "d:\Combi_annotations\All_Annotations"
    destination_directory = "d:\Combi_annotations\combined_dataset"

    copy_files(source_directory, destination_directory)
