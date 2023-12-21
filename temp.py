import os
import filecmp
import shutil

def remove_duplicates(source_folder, comparison_folder):
    # Create a list of files in each folder
    source_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    comparison_files = [f for f in os.listdir(comparison_folder) if os.path.isfile(os.path.join(comparison_folder, f))]

    # Iterate over files in the source folder
    for source_file in source_files:
        source_file_path = os.path.join(source_folder, source_file)

        # Check if the file exists in the comparison folder
        duplicate = False
        for comparison_file in comparison_files:
            comparison_file_path = os.path.join(comparison_folder, comparison_file)
            if filecmp.cmp(source_file_path, comparison_file_path, shallow=False):
                duplicate = True
                break

        # If the file is a duplicate, delete it
        if duplicate:
            # os.remove(source_file_path)
            print(f"Deleted duplicate file: {source_file}")

if __name__ == "__main__":
    source_folder = "/media/harsha/New Volume1/Harsha/Front_Annotated_Dataset/logos/YOLODataset/images/val"
    comparison_folder = "/media/harsha/New Volume1/Harsha/Front_Annotated_Dataset/logos/YOLODataset/labels/val"

    if not os.path.exists(source_folder) or not os.path.exists(comparison_folder):
        print("One or both of the specified folders do not exist.")
    else:
        remove_duplicates(source_folder, comparison_folder)
        print("Duplicate files removed.")



# # import os

# # folder_path = "/media/harsha/New Volume1/Harsha/Front_Annotated_Dataset/logos/YOLODataset/labels/train"  # Replace with the path to your folder

# # # Check if the folder exists
# # if not os.path.exists(folder_path):
# #     print("Folder does not exist.")
# # else:
# #     for filename in os.listdir(folder_path):
# #         if "front" in filename:
# #             file_path = os.path.join(folder_path, filename)
# #             if os.path.isfile(file_path):
# #                 os.remove(file_path)
# #                 print(f"Deleted file: {filename}")

# # print("Deletion of files with 'front' in the name is complete.")
# import os
# import shutil

# def copy_matching_files(input_folder, search_folder1, search_folder2, output_folder):
#     if not os.path.exists(input_folder) or not os.path.exists(search_folder1) or not os.path.exists(search_folder2):
#         print("One or more folders do not exist.")
#         return

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Get the list of JPG files in the input folder
#     jpg_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]

#     for jpg_file in jpg_files:
#         jpg_filename, _ = os.path.splitext(jpg_file)
        
#         # Search for matching TXT files in both search folders
#         matching_txt_file1 = os.path.join(search_folder1, jpg_filename + ".txt")
#         matching_txt_file2 = os.path.join(search_folder2, jpg_filename + ".txt")

#         # Check if the matching TXT file exists in the first search folder
#         if os.path.exists(matching_txt_file1):
#             shutil.copy(matching_txt_file1, output_folder)
#             print(f"Copied {matching_txt_file1} to {output_folder}")

#         # Check if the matching TXT file exists in the second search folder
#         if os.path.exists(matching_txt_file2):
#             shutil.copy(matching_txt_file2, output_folder)
#             print(f"Copied {matching_txt_file2} to {output_folder}")

# if __name__ == "__main__":
#     input_folder = "/media/harsha/New Volume1/Harsha/Front_Annotated_Dataset/logos/YOLODataset/images/train"
#     search_folder1 = "/media/harsha/New Volume1/Harsha/Front_Annotated_Dataset/logos/YOLODataset/labels/front"
#     search_folder2 = "/media/harsha/New Volume1/Harsha/Front_Annotated_Dataset/logos/YOLODataset/labels/rear"
#     output_folder = "/media/harsha/New Volume1/Harsha/Front_Annotated_Dataset/logos/YOLODataset/labels/train"

#     copy_matching_files(input_folder, search_folder1, search_folder2, output_folder)
