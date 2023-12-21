import os
import json
from PIL import Image, ImageDraw
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import glob

def extract_segments_from_json(folder_path, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            json_path = os.path.join(folder_path, filename)
            
            # Load the JSON file
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                
                # Get the image file name and path
                image_filename = data['imagePath']
                image_path = os.path.join(folder_path, image_filename)
                
                # Load the image
                image = Image.open(image_path)
                
                # Iterate through each object in the JSON file
                for obj in data['shapes']:
                    class_name = obj['label']
                    points = obj['points']
                    
                    # Convert points to a numpy array
                    points = np.array(points, dtype=np.int32)
                    
                    # Create a subfolder for the class if it doesn't exist
                    class_folder = os.path.join(output_folder, class_name)
                    os.makedirs(class_folder, exist_ok=True)
                    
                    # Create a mask for the segment
                    mask = Image.new("RGB", image.size, 0)
                    mask_array = np.zeros_like(np.array(mask),  dtype=np.uint8)
                    ImageDraw.Draw(mask).polygon(tuple(map(tuple, points)), outline=1, fill=1)
                    mask_array += np.array(mask)
                    
                    # Apply the mask to the image and save as segment
                    segment = Image.fromarray(np.uint8(np.array(image) * mask_array))
                    segment_filename = os.path.splitext(image_filename)[0] + '_' + class_name + '.png'
                    segment_path = os.path.join(class_folder, segment_filename)
                    segment.save(segment_path)
                    print(segment_path)

def extract_segments_from_xml(folder_path, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            xml_path = os.path.join(folder_path, filename)

        #try:
            # parse the content of the xml file
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            image_path = os.path.join(folder_path, filename.split('.xml')[0] + ".png")
            # image_path = root.find('path').text
            
            # Load the image
            print('path',image_path)
            image = cv2.imread(image_path)

            for obj in root.findall('object'):
                # Get class name
                class_name = obj.find("name").text

                # Get bounding box coordinates
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)
                
                # Create a subfolder for the class if it doesn't exist
                class_folder = os.path.join(output_folder, class_name)
                os.makedirs(class_folder, exist_ok=True)
                
                box_segment = image[ymin:ymax, xmin:xmax, :]

                # Save box image segment
                output_filename = f"{class_name}_{filename[:-4]}.jpg"
                output_path = os.path.join(class_folder, output_filename)
                cv2.imwrite(output_path, box_segment)

                # # Create a mask for the segment
                # mask = Image.new("RGB", image.size, 0)
                # mask_array = np.zeros_like(np.array(mask),  dtype=np.uint8)
                # ImageDraw.Draw(mask).polygon(tuple(map(tuple, points)), outline=1, fill=1)
                # mask_array += np.array(mask)
                
                # # Apply the mask to the image and save as segment
                # segment = Image.fromarray(np.uint8(np.array(image) * mask_array))
                # segment_filename = os.path.splitext(image_filename)[0] + '_' + class_name + '.png'
                # segment_path = os.path.join(class_folder, segment_filename)
                # segment.save(segment_path)
                # print(segment_path)


                # class_names_list.append(label)
        # except:
        #     print("xml file name having issue: ", filename)
        


if __name__ == '__main__':
    input_folders_path = r"D:\Combi_annotations\Annotations\Zuber\CT"
    # folder_path = "/media/harsha/New Volume/Harsha_Projects/CMPV_Data/front_with_foglamp/Hello"
    output_folder = r"D:\Combi_annotations\Segments\Zuber"
    # extract_segments_from_json(folder_path, output_folder)
    
    input_folders_path_list = glob.glob(input_folders_path, recursive=True)
    idx = 0
    input_folders_path_list = os.listdir(input_folders_path)
    for folder_name in input_folders_path_list:
        if folder_name.endswith('.avi'):
            continue
        folder_path = os.path.join(input_folders_path, folder_name)
        print("folder path  :", folder_path)
        extract_segments_from_xml(folder_path, output_folder)