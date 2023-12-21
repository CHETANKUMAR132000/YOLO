import os
import random
import shutil

def random_select_images(input_folder, output_folder, num_images=100):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    all_images = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # Randomly select num_images from the list
    selected_images = random.sample(all_images, min(num_images, len(all_images)))

    # Copy the selected images to the output folder
    for image in selected_images:
        src_path = os.path.join(input_folder, image)
        dest_path = os.path.join(output_folder, image)
        shutil.copy2(src_path, dest_path)

if __name__ == "__main__":
    # Replace 'input_folder' and 'output_folder' with your actual folder paths
    input_folder = '/media/harsha/New Volume/Harsha_Projects/logo_error_Data/logo_error_classes_data/frames'
    output_folder = '/media/harsha/New Volume/Harsha_Projects/logo_error_Data/logo_error_classes_data/frames_sampled'

    # Specify the number of images to randomly select
    num_images_to_select = 100

    random_select_images(input_folder, output_folder, num_images_to_select)
    print(f"{num_images_to_select} images randomly selected and copied to {output_folder}")
