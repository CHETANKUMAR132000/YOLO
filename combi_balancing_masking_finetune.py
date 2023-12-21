import os
import re
import shutil
import glob
import random
import xml.etree.ElementTree as ET
import json
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import OrderedDict
import chardet

def read_labels_file_return_labels(labels_file):
    labels = []
    # Detect the encoding of the file
    with open(labels_file, "rb") as f:
        rawdata = f.read()
        result = chardet.detect(rawdata)
        file_encoding = result['encoding']

    with open(labels_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label = line.split()[0]
            if label:
                labels.append(label)
    f.close()
    return labels

def read_xml_file(xml_label_file):
    """
    """
    class_names_list = []
    
    try:
        # parse the content of the xml file
        tree = ET.parse(xml_label_file)
        root = tree.getroot()

        for obj in root.findall('object'):
            label = obj.find("name").text
            class_names_list.append(label)
    except:
        print("xml file name having issue: ", xml_label_file)
    return class_names_list

def read_xml_count_classes(xml_files_dir):
    # global total_labels_count_dict
    total_labels_count_dict = {}
    # identify all the xml files in the annotations folder (input directory)
    files = glob.glob(os.path.join(xml_files_dir, '*.xml'))
    # loop through each 
    file_count = 0
    for xml_label_file in files:
        class_names_list = read_xml_file(xml_label_file)
        for cls_name in class_names_list:
            if cls_name not in total_labels_count_dict.keys():
                total_labels_count_dict[cls_name] = 1
            else:
                total_labels_count_dict[cls_name] += 1
        file_count += 1
        # print("File num processing:", file_count)

            # if int(label) > 32: # handling 32+1 classes currently, label starts from 0 to 32
            #     pass
            # else:
                # cls_name = labels_map_dict[int(label)]
    return total_labels_count_dict

def swap_keys_values(dictionary):
    return {value: key for key, value in dictionary.items()}

def read_combi_classes(cls_filename):
    items = []
    lut = {}

    with open(cls_filename, 'r') as file:
        content = file.read()
        
        # Extract item blocks using regular expression
        # item_blocks = re.findall(r'item {[/s/S]*?}', content)
        item_blocks = re.findall(r'item {([^}]*)}', content, re.DOTALL)
        # print(item_blocks)

        
        for item_block in item_blocks:
            item = {}
            
            # Extract id and name using regular expression
            # id_match = re.search(r'id: (/d+)', item_block)
            id_match = re.search(r"id:\s*(\d+)", item_block)
            name_match = re.search(r"name: '(.+)'", item_block)

            if id_match and name_match:
                
                item['id'] = int(id_match.group(1))
                item['name'] = name_match.group(1)
                lut[item['name']] = item['id'] -1   #in class file , num start from 1, yolo requires from 0
                if item['name'] not in items:
                    items.append(item['name'])
    # print(lut)
    return lut

def xml_to_yolo_bbox(bbox, w, h):
    # bbox = [xmin, ymin, xmax, ymax]
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

def xml_to_yolo_segment(bbox, w, h):
    # xmin, ymin, xmax, ymax
    xmin = bbox[0] / w
    ymin = bbox[1] / h
    xmax = bbox[2]/ w
    ymax = bbox[3] / h
    return [xmin, ymin, xmax, ymax]
  
def xml2yolo_txt(xml_input_dir, output_dir, image_dir, cls_filename):
    class_map_dict = read_combi_classes(cls_filename)
    # class_map_dict['fort_power_mode_indicator_light'] = 33
    # class_map_dict['fort_eco_mode_indicator_light'] = 34

    # create the labels folder (output directory)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # identify all the xml files in the annotations folder (input directory)
    files = glob.glob(os.path.join(xml_input_dir, '*.xml'))
    count = 0
    for fil in files:
        basename = os.path.basename(fil)
        filename = os.path.splitext(basename)[0]

        result = []

        # parse the content of the xml file
        try:
            tree = ET.parse(fil)
            root = tree.getroot()
            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)

            for obj in root.findall('object'):
                label = obj.find("name").text

                if label == "imt_indicator_light":
                    label = "imt_green"
                if label == "sport_mode_indicator":
                    label = "sport_mode_indicator_light"
                if label == "pbk_indicator_light":
                    label = "pkb_indicator_light"
      
                if label in class_map_dict:
                    label_str = str(class_map_dict[label]) 
                    pil_bbox = [int(x.text) for x in obj.find("bndbox")]
                    yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
                    # yolo_segm = xml_to_yolo_segment(pil_bbox, width, height)
                    # convert data to string
                    bbox_string = " ".join([str(x) for x in yolo_bbox])
                    result.append(f'{label_str} {bbox_string}')

                else:
                    print("class label/name is not in list: ", label)


            if result:
                # generate a YOLO format text file for each xml file
                with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
                    f.write('\n'.join(result) + '\n')
                    # f.write('/n'.join(result))
            else:
                count += 1
        except:
            # handling empty xml files : M51250194 series are empty
            print("xml parsing error in filename: ", fil)
            count += 1
            # del root, tree

    print("num of images skipped: ", count)

def check_common_elements(list1, list2):
    return any(element in list2 for element in list1)

def fetch_cls_enough_data(cls_req_dict, min_instances):
    remove_key = None
    for key, value in cls_req_dict.items():
        if value >= min_instances:
            remove_key = key  
    return remove_key

def check_min_count(labels_count_dict, min_instances):
    # print("label counts: ", labels_count_dict)
    temp_dict = labels_count_dict.copy()
    # print("Printing temp dict:", temp_dict)
    labels_list = list(labels_count_dict.keys())
    remove_cls_list = ['urea_scr_adblue_warning_light', 'power_mode_indicator_light', 'eco_mode_indicator_light']
    if check_common_elements(remove_cls_list, labels_list):
        del temp_dict['urea_scr_adblue_warning_light']
        del temp_dict['power_mode_indicator_light']     #temporarily remove the keys as these classes not present in current dataset xml files
        del temp_dict['eco_mode_indicator_light']
    is_min_count = all(int(value) >= min_instances for value in temp_dict.values())
    del temp_dict
    return is_min_count

def filter_images_by_label(images_folder, label_file, output_folder, labels_map_dict, labels_count_dict):
    global total_imgs_count
    
    labels = read_labels_file_return_labels(label_file_path)
    # print(labels_map_dict)
    for label in labels:
        # if int(label) > 32: # handling 32+1 classes currently, label starts from 0 to 32
        #     pass
        # else:
        cls_name = labels_map_dict[int(label)]
        labels_count_dict[cls_name] += 1

    label_file_name = os.path.basename(label_file_path)
    img_path = os.path.join(images_folder, label_file_name.split('.')[0] + '.jpg')

    if os.path.exists(label_file_path) and os.path.exists(img_path):
        shutil.copy(label_file, output_folder)
        shutil.copy(img_path, output_folder)
        total_imgs_count += 1
    return labels_count_dict

def fetch_specific_class(ip_images_folder, ip_txt_files_dir, labels_map_dict, cls_req_list, labels_count_dict, min_instances, op_save_folder):
    """
    given img folder and txt labels folder, 
    filter data according the condition set
    condition: only look for txt files which have some minority class present in it as by default the majority classes will also get covered in it
    copy the filtered imgs+txt labels into a different folder

    args:
    ip_images_folder : input images folder having the full dataset
    ip_txt_files_dir : input labels/txt/yolo format folder having the full dataset
    cls_req_list : list of class names which are to be sampled from the larger dataset
    labels_count_dict : overall classes present
    op_save_folder : folder where to move the filtered images and labels

    """
    print("Fetching specific classes")
    total_imgs_count = 0
    copied_img_list = []
    txt_file_list = os.listdir(ip_txt_files_dir)
    random.shuffle(txt_file_list)

    cls_req_dict =  dict.fromkeys(cls_req_list, 0)
   
    
    for label_file in txt_file_list:
        label_file_path = os.path.join(ip_txt_files_dir, label_file )
        labels = read_labels_file_return_labels(label_file_path)
        #convert label_nums into class names from map
        classes_present_list = []
        for label in labels:
            classes_present_list.append(labels_map_dict[int(label)])
        # classes_present_list.append(labels_map_dict[label] for label in labels)
        
        #implemented to get bare minimum data, this is not required in initial data sampling for first time training as in that case minority classes will have 
        #only min_instances samples and other classes may be in large number
        remove_key = fetch_cls_enough_data(cls_req_dict, min_instances)
        if remove_key:
            cls_req_list.remove(remove_key)
            cls_req_dict.pop(remove_key)


        #processing only text files which have minorty classes in it
        if check_common_elements( cls_req_list, classes_present_list):
            # print(classes_present_list)
            for label in labels:
                cls_name = labels_map_dict[int(label)]
                labels_count_dict[cls_name] += 1
                if cls_name in cls_req_list:
                    cls_req_dict[cls_name] += 1

            label_file_name = os.path.basename(label_file_path)
            img_path = os.path.join(ip_images_folder, label_file_name.split('.txt')[0] + '.png')

            if os.path.exists(label_file_path) and os.path.exists(img_path):
                shutil.copy(label_file_path, op_save_folder)
                shutil.copy(img_path, op_save_folder)
                copied_img_list.append(img_path)
                total_imgs_count += 1    
                # labels_count_dict = filter_images_by_label(images_folder, label_file_path, filtered_min_data_folder, labels_map_dict, labels_count_dict)
                print("Total images copied:", total_imgs_count)
            # print("count dict:", labels_count_dict)
        
        else:
            print("Required minority classes not present in filename: ", label_file_path)
            pass
        
        is_min_count = check_min_count(cls_req_dict, min_instances)
        if is_min_count: # if all classes have got min_instances
            print("Got enough images : End operation") 
            return labels_count_dict, copied_img_list
        else:
            continue
    return labels_count_dict, copied_img_list

def fetch_cls_not_in_list(images_folder, txt_output_dir, copied_img_list, op_save_folder):
    if not os.path.isdir(op_save_folder):
        os.mkdir(op_save_folder)
    img_file_list = os.listdir(images_folder)
    img_file_list = [os.path.join(images_folder, img_name) for img_name in img_file_list]
    left_img_list = set(img_file_list) -  set(copied_img_list)
    for img_path in left_img_list:
        label_file_name =  os.path.basename(img_path).split(".png")[0]
        label_file_path = os.path.join(txt_output_dir, label_file_name + ".txt")
        shutil.copy(label_file_path, op_save_folder)
        shutil.copy(img_path, op_save_folder)


def minority_class_sample_and_move(images_folder, txt_files_dir, labels_map_dict, overall_minority_cls_list, labels_count_dict, min_instances, filtered_min_data_folder):
    """
    given img folder and txt labels folder, 
    filter data according the condition set
    copy the filtered imgs+txt labels into a different folder
    """
    total_imgs_count = 0
    txt_file_list = os.listdir(txt_files_dir)
    random.shuffle(txt_file_list)
    
    for label_file in txt_file_list:
        label_file_path = os.path.join(txt_files_dir, label_file )
        labels = read_labels_file_return_labels(label_file_path)
        #convert label_nums into class names from map
        classes_present_list = []
        for label in labels:
            classes_present_list.append(labels_map_dict[int(label)])
        # classes_present_list.append(labels_map_dict[label] for label in labels)
        if check_common_elements( overall_minority_cls_list, classes_present_list):
            # print(classes_present_list)
            for label in labels:
                cls_name = labels_map_dict[int(label)]
                labels_count_dict[cls_name] += 1

            label_file_name = os.path.basename(label_file_path)
            img_path = os.path.join(images_folder, label_file_name.split('.')[0] + '.jpg')

            if os.path.exists(label_file_path) and os.path.exists(img_path):
                shutil.copy(label_file_path, filtered_min_data_folder)
                shutil.copy(img_path, filtered_min_data_folder)
                total_imgs_count += 1    
                # labels_count_dict = filter_images_by_label(images_folder, label_file_path, filtered_min_data_folder, labels_map_dict, labels_count_dict)
                print("Total images copied:", total_imgs_count)
            # print("count dict:", labels_count_dict)
        
        else:
            # print("Minority classes not present in filename: ", label_file_path)
            pass
        
        is_min_count = check_min_count(labels_count_dict, min_instances)
        if is_min_count: # if all classes have got min_instances
            print("Got enough images : End operation") 
            return labels_count_dict 
        else:
            continue
    return labels_count_dict

def combi_results_generate():
    pass
    
def total_class_count_from_txt(labels_folder, total_labels_count_dict, num_labels_map_dict_highcross):
    # total_labels_count_dict = {}
    counter = 0
    for label_file in os.listdir(labels_folder):
        # is_min_count = check_min_count(labels_count_dict, min_instances)
        label_file_path = os.path.join(labels_folder, label_file )
        counter += 1
        # print("Processing file num:", counter)
        labels = read_labels_file_return_labels(label_file_path)
        for label in labels:
            cls_name = num_labels_map_dict_highcross[int(label)]
            if cls_name in cls_name_number_map_dict.keys():
                total_labels_count_dict[cls_name] += 1
            else:
                total_labels_count_dict[cls_name] = 1
                # print("class not present in txt files:", label)

    return total_labels_count_dict

def total_class_count_from_txt_folder(labels_folder):
    total_labels_count_dict = {}
    counter = 0
    for label_file in os.listdir(labels_folder):
        # is_min_count = check_min_count(labels_count_dict, min_instances)
        label_file_path = os.path.join(labels_folder, label_file )
        counter += 1
        # print("Processing file num:", counter)
        labels = read_labels_file_return_labels(label_file_path)
        for label in labels:
            if label in total_labels_count_dict:
                total_labels_count_dict[label] += 1
            else:
                total_labels_count_dict[label] = 1

    return total_labels_count_dict

def plot_class_disribution(class_dict, plot_name, plt_save_dir):
    
    sorted_dict = {k: v for k, v in sorted(class_dict.items(), key=lambda item: item[1], reverse=True)}

    class_names = list(sorted_dict.keys())
    class_values = list(sorted_dict.values())

    fig, ax = plt.subplots()

    # Create a bar plot using seaborn
    sns.barplot(x=class_names, y=class_values, ax=ax)

    # Add labels and title
    plt.xlabel('Class Names')
    plt.ylabel('Values')
    plt.title('Class Distribution')

    # Set the resolution of the saved image
    dpi = 300

    plt.xticks(rotation=45, ha='right', fontsize=6)
      # Add values to the bars
    for i, (value, name) in enumerate(zip(class_values, class_names)):
        ax.text(i, value, str(value), ha='center', va='bottom', fontsize=6)
    

    plt_save_path = os.path.join(plt_save_dir, plot_name + '.jpg')
    # Save the plot as a PNG image with high resolution
    plt.savefig(plt_save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

def custom_mask_func(img, mask):
    """
    from centre of mask till boundary-1 pixel : make it same color as background
    boundary pixel value : interpolate values from the surrounding region to maintain a consistent transition to outer regions

    Task: how to find out what exactly are background values : avg color in hsv for whole image?
    """
    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour (which should be the border)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the border region
    border_mask = np.zeros_like(mask)
    cv2.drawContours(border_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Interpolate the border pixels from the surrounding pixels
    result = img.copy()

    inpaint_results = cv2.inpaint(img, border_mask, 3, cv2.INPAINT_NS)
    # result[border_mask == 255] = inpaint_results

    return inpaint_results


def mask_all_except_common_cls(image_path, boxes, common_class, mask_type):
    # Load the image
    image = cv2.imread(image_path)
    
    # Create a copy of the image for masking
    image_copy = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Iterate through each box
    for label, x, y, w, h in boxes:
        if int(label) != common_class:
            # Get the surrounding region of the box
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            # Scale the box coordinates back to the image size
            x_centre = int(x * image.shape[1])
            y_centre = int(y * image.shape[0])
            w = int(w * image.shape[1])
            h = int(h * image.shape[0])

            x = int(x_centre - w/2)
            y = int(y_centre - h/2)
            
            #adding few pixels margin to see its effect on masking algorithms
            # padding_pixels = 7
            # x -= padding_pixels
            # y -= padding_pixels
            # w += padding_pixels
            # h += padding_pixels

            # mask_region = image[y:y+h, x:x+w]
 
            # Set the region inside the box as white in the mask
            mask[y:y + h, x:x + w] = 255
            
            
            if mask_type == 'black':
                image_copy[y:y + h, x:x + w, :] = 0  # set as black   
            elif mask_type == 'interpolate':
                # Perform interpolation on the surrounding region
                image_copy = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
                # image_copy = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
            elif mask_type == 'custom':
                # image_copy = custom_mask_func(image_copy, mask)
                pass
            print("mask genereated for label: ", label)
            
    return image_copy

def mask_image_given_cls_list(image_path, boxes, labels_to_mask_list, mask_type):
    # Load the image
    image = cv2.imread(image_path)
    
    # Create a copy of the image for masking
    image_copy = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Iterate through each box
    for label, x, y, w, h in boxes:
        if int(label) in labels_to_mask_list:
            # Get the surrounding region of the box
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            # Scale the box coordinates back to the image size
            x_centre = int(x * image.shape[1])
            y_centre = int(y * image.shape[0])
            w = int(w * image.shape[1])
            h = int(h * image.shape[0])

            x = int(x_centre - w/2)
            y = int(y_centre - h/2)
            
            #adding few pixels margin to see its effect on masking algorithms
            # padding_pixels = 7
            # x -= padding_pixels
            # y -= padding_pixels
            # w += padding_pixels
            # h += padding_pixels

            # mask_region = image[y:y+h, x:x+w]
 
            # Set the region inside the box as white in the mask
            mask[y:y + h, x:x + w] = 255
            
            
            if mask_type == 'black':
                image_copy[y:y + h, x:x + w, :] = 0  # set as black   
            elif mask_type == 'interpolate':
                # Perform interpolation on the surrounding region
                image_copy = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
                # image_copy = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
            elif mask_type == 'custom':
                # image_copy = custom_mask_func(image_copy, mask)
                pass
            
            # print("mask genereated for label: ", label)
            
    return image_copy


# def create_new_label_file(op_txt_path, boxes_list, common_class_label_num_list):
#     for label, x, y, w, h in boxes_list:   
        
#         if int(label) in common_class_label_num_list:
#             data = [label, x, y, w, h]
#             with open(op_txt_path, 'w') as f:
#                 line = ' '.join(data)
#                 f.write(line)


def create_new_label_file(op_txt_path, boxes_list, common_class_label_num_list):
    with open(op_txt_path, 'w') as f:
        for label, x, y, w, h in boxes_list:   
            if int(label) in common_class_label_num_list:  
                data = [label, x, y, w, h]
                line = ' '.join(map(str, data))  # Convert all elements to strings before joining
                f.write(line + '\n')  # Add a newline after writing each da


def convert_num_into_classnames(labels_list, num_labels_map_dict):
    class_names_list = []
    for i in labels_list:
        class_names_list.append(num_labels_map_dict[int(i)])
    return class_names_list

def mask_allclasses_except1_sampling(req_class_list, min_instances, img_folder_path, label_file_folder,  num_labels_map_dict, cls_name_number_map_dict, op_save_path):
    '''
    1. get list of classes to be obtained 
    2. read the master directory which has all images
    3. for every image, read txt file corresponding to it with same name
    4. if required class label in the list of labels, process this txt and image pair
    5. create a dict with classes to be masked i.e all classes except te required class
        5.1 dict to have key as class name and value as 4 box coordinates
    6. Iteratively go to each box and fill it with interpolated values from its neighbourhood eg: INTER_Bilienar interpolation
    7. remove all boxes from the txt file except the current required class being processed
    8. save a copy of the edited image and edited txt file in destination folder
    8. update required class list with count and remove 
    '''

    # img_list = os.listdir(img_folder_path)
    img_list = [file for file in os.listdir(img_folder_path) if file.endswith(".png")]
    random.shuffle(img_list)

    req_class_dict =  dict.fromkeys(req_class_list, 0)
 
    for img_path in img_list:
        file_name = os.path.basename(img_path)
        label_file_path = os.path.join(label_file_folder, file_name.split('.png')[0] + '.txt')
        if os.path.exists(label_file_path):
            labels_list = read_labels_file_return_labels(label_file_path)
        else:
            continue # skip image as txt file doesnt exit (because xml itself is not there when image is not having any labels)        
        
        # if required class label in the list of labels, process this txt and image pair
        class_names_list = convert_num_into_classnames(labels_list, num_labels_map_dict)
        
        # update req_class_list as only min_instances are required
        remove_key = fetch_cls_enough_data(req_class_dict, min_instances)
        if remove_key:
            req_class_list.remove(remove_key)
            req_class_dict.pop(remove_key)
        # check if all classes fetched
        if req_class_list == []:
            print("Masking operation successfull and all classes fetched")
            return 0
        else:
            pass

        if check_common_elements(req_class_list, class_names_list):
            mask_classes_boxes_dict = {}
            
            common_class_name = [element for element in class_names_list if element in req_class_list ][0] # get only 1 class
            mask_cls_list =  [x for x in class_names_list if x not in [common_class_name]]
            common_class_label_num = cls_name_number_map_dict[common_class_name]
            for cls in mask_cls_list:
                mask_classes_boxes_dict[cls] = [] # placeholder for actual box coordinates
            
            boxes_list = []
            # open text file to get coordinates of annotatd boxes
            # currently writing a non-optimal way of iterating over files - O(n*n) time complexity
            with open(label_file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    cls_label, x, y, w, h = line.split()
                    boxes_list.append([cls_label, x, y, w, h])

                    # for cls_ in mask_cls_list:
                    #     masking_cls_label = cls_name_number_map_dict[cls_]
                    #     if masking_cls_label == cls_label: #need to test this and modify 
                    #         mask_classes_boxes_dict[cls_] = [x, y, h, w]
            # mask image
            img_path = os.path.join(img_folder_path, img_path)
            masked_image = mask_all_except_common_cls(img_path, boxes_list, common_class_label_num, mask_type='black')
            img_name = os.path.basename(img_path)
            
            op_img_path = os.path.join(op_save_path, img_name)
            cv2.imwrite(op_img_path, masked_image)

            # create new text label file with just the unmasked class data
            txt_file_name = img_name.split('.png')[0] + '.txt'
            op_txt_path = os.path.join(op_save_path, txt_file_name)
            common_class_label_num_list = [common_class_label_num]
            create_new_label_file(op_txt_path, boxes_list, common_class_label_num_list)
            
            # update req_class_list if enough count images are received 
            req_class_dict[common_class_name] += 1
            print("Class count: ", req_class_dict)
            
    return 0

def focal_loss_function():
    pass

def balance_classes_by_masking(req_class_list, req_count, img_folder_path, label_file_folder, num_labels_map_dict,cls_name_number_map_dict, op_save_path, mask_type):
    """
    given the data distribution dict, 
    initially fetch images with all labels, 
    update labels received dict
    check if enough instances of any classs received: if yes, add the class name to mask_class_list
    use this list to mask the current image and its txt/label file
    save both in new folder
    Inputs :
    Outputs : 
    """
    if not os.path.isdir(op_save_path):
        os.mkdir(op_save_path)
    class_req_dict = {}
    # ip_class_list = req_class_list
    
    img_path_list = [file for file in os.listdir(img_folder_path) if file.endswith(".png")]
    random.shuffle(img_path_list)

    class_req_dict =  dict.fromkeys(req_class_list, 0)
    mask_class_list = []
 
    for img_path in img_path_list:
  
        file_name = os.path.basename(img_path)
        label_file_path = os.path.join(label_file_folder, file_name.split('.png')[0] + '.txt')
        if os.path.exists(label_file_path):
            labels_list = read_labels_file_return_labels(label_file_path)
        else:
            continue # skip image as txt file doesnt exit (because xml itself is not there when image is not having any labels)      
        
        # labels_list = read_labels_file_return_labels(label_file_path)
        
        # if required class label in the list of labels, process this txt and image pair
        class_names_list = convert_num_into_classnames(labels_list, num_labels_map_dict)
        
        # if enough instance of a class received, then add it to mask list
        for key, value in class_req_dict.items():
            if value >= req_count:
                if key not in mask_class_list:
                    mask_class_list.append(key)
        
        mask_class_numbers_list = [] 
        for cls_name in mask_class_list:
            mask_class_numbers_list.append(cls_name_number_map_dict[cls_name])
        
        int_labels_list = [int(item) for item in labels_list]
        
        if len(mask_class_list) == len(class_req_dict.keys()):
            print("Required instances of all classes achieved and cant mask all classes as it will make an empty image/label fil")
            return 0
        # check if labels list is a subset of mask list (i.e if i use mask_list, it will mask all the classes in current imaage)
        if set(int_labels_list).issubset(set(mask_class_numbers_list)):
            print("skipped image: ", img_path)
            continue  # move to next image, so that we dont have an empty label file/txt file
        else:
            pass  


        boxes_list = []
            # open text file to get coordinates of annotatd boxes
            # currently writing a non-optimal way of iterating over files - O(n*n) time complexity
        with open(label_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                cls_label, x, y, w, h = line.split()
                boxes_list.append([cls_label, x, y, w, h])

            # mask image
        img_path = os.path.join(img_folder_path, img_path)  
        img_name = os.path.basename(img_path)
        print("Mask list: ", mask_class_list)    
        if mask_class_list:

            masked_image = mask_image_given_cls_list(img_path, boxes_list, mask_class_numbers_list, mask_type=mask_type)
            op_img_path = os.path.join(op_save_path, img_name)
            cv2.imwrite(op_img_path, masked_image)
            print("Image masked : ", op_img_path)
            
            # create new text label file with just the unmasked class data
            txt_file_name = img_name.split('.png')[0] + '.txt'
            op_txt_path = os.path.join(op_save_path, txt_file_name)
            non_mask_class_list = set(req_class_list) - set(mask_class_list)
            non_mask_class_numbers_list = []
            
            for cls_name in non_mask_class_list:
                non_mask_class_numbers_list.append(cls_name_number_map_dict[cls_name])

            create_new_label_file(op_txt_path, boxes_list, non_mask_class_numbers_list)
        else:
            shutil.copy(img_path, op_save_path)
            shutil.copy(label_file_path, op_save_path)

            # update classes obatined till now
        for cls_name in class_names_list:
            class_req_dict[cls_name] += 1
        # print("Class count: ", class_req_dict)      

     
    return 0

def mask_all_test_annotations(img_folder_path, label_file_folder, num_labels_map_dict_highcross, op_save_path):
    """
    mask all the annotations in the image with black pixels 
    do for all images
    Result: by visual inspection, if any class was not annotated would be seen in masked images
    """
    if not os.path.isdir(op_save_path):
        os.mkdir(op_save_path)
    img_path_list = [file for file in os.listdir(img_folder_path) if file.endswith(".png")]

    for img_path in img_path_list:
  
        file_name = os.path.basename(img_path)
        label_file_path = os.path.join(label_file_folder, file_name.split('.png')[0] + '.txt')
        if os.path.exists(label_file_path):
            labels_list = read_labels_file_return_labels(label_file_path)
        else:
            continue # skip image as txt file doesnt exit (because xml itself is not there when image is not having any labels)      

        boxes_list = []
            # open text file to get coordinates of annotatd boxes
            # currently writing a non-optimal way of iterating over files - O(n*n) time complexity
        with open(label_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                cls_label, x, y, w, h = line.split()
                boxes_list.append([cls_label, x, y, w, h])
        
        
        img_path = os.path.join(img_folder_path, img_path)  
        img_name = os.path.basename(img_path)
        # print("Mask list: ", mask_class_list)    
        labels_to_mask_list = list(num_labels_map_dict_highcross.keys())
        masked_image = mask_image_given_cls_list(img_path, boxes_list, labels_to_mask_list, mask_type='black')
        op_img_path = os.path.join(op_save_path, img_name)
        cv2.imwrite(op_img_path, masked_image)
        print("Image masked : ", op_img_path)   
    print("Masking all Images complete!")
    return 0




if __name__ == '__main__':
    images_folder = 'd:\Combi_annotations\combined_dataset'

    xml_files_dir = "d:\Combi_annotations\combined_dataset"
    txt_output_dir = "d:\Combi_annotations\combined_dataset/txts"
    
    cls_filename = "d:\Combi_annotations/combi_cmpv.pbtxt"
    # rear_cls_filename = 'D:/highcross/highcross_data/rear_highcross_classes.pbtxt'
    # filtered_min_data_folder = 'D:/highcross/highcross_data/combi_2/finetune_v1_10'
    plt_save_dir =  'd:\Combi_annotations'
    labels_count_dict = {}
    # num_labels_map_dict = {0: 'master_caution_warning_light', 1: 'brake_warning_Light',2: 'pkb_indicator_light',3: 'seat_belt_warning_light',4: 'air_bag_warning_light',5: 'urea_scr_adblue_warning_light',6: 'check_engine_warning_light',7: 'slip_vsc_indicator_light',8: 'slip_off_vsc_off_indicator',9: 'abs_indicator_light',10: 'fuel_Level_warning_light',11: 'glow_plug_warning_light',12: 'rear_fog_Lamp_Indicator',13: 'cruise_control_indicator_light_Green',14: 'high_beam_indicator_light',15: 'tail_lamp_position_indicator_light',16: 'front_fog_lamp_indicator_light',17: 'left_turn_signal_indicator_light',18: 'right_turn_signal_indicator_light',19: 'power_mode_indicator_light',20: 'eco_mode_indicator_light',21: 'eco_indicator_light',22: 'rr_diff_lock',23: '4_lo_mode_indication',24: 'vfc_indicator',25: 'auto_lsd_indicator_light',26: 'dac_system_light_indicator',27: '4wd_system_indication',28: 'imt_green',29: 'eco_run_off',30: 'eco_run_on',31: 'DPF',32: 'sport_mode_indicator_light',33: 'fort_power_mode_indicator_light', 34: 'fort_eco_mode_indicator_light'}
    # num_labels_map_dict_highcross = {0: 'seat_belt_warning_light',1: 'air_bag_warning_light',2: 'fuel_Level_warning_light',3: 'check_engine_warning_light',4: 'power_off_light',5: 'vfc_indicator',6: 'Brake_hold',7: 'abs_indicator_light',8: 'slip_vsc_indicator_light',9: 'pkb_indicator_light',10: 'Brake_hold_standby',11: 'brake_warning_Light_without_ABS',12: 'brake_warning_Light_with_ABS',13: 'rear_seat',14: 'sunroof',15: 'ABS_off',16: 'tail_lamp_position_indicator_light',17: 'high_beam_indicator_light',18: 'slip_vsc_indicator_light_on',19: 'airbag_allert',20: 'accident_alert',21: 'highway_drive',22: 'road_view',23: 'ready_to_drive',24: 'right_turn_signal_indicator_light',25: 'left_turn_signal_indicator_light', 26: 'fuel_Level_warning_light_2', 27: 'ev_indicator', 28: 'front_fog_lamp_indicator_light', 29: 'rr_diff_lock', 30: 'lock', 31: 'eco_indicator_1', 32: 'eco_indicator_2', 33: 'sunroof',34: 'power_off_light',35: 'Temperature_warning', 36: 'Drive_start_control', 37: 'beam'}
    # num_labels_map_dict_highcross_rear = {0:'Sharkfin_Antenna',1: 'Rear_Spoiler',2: 'High_mount_stop_lamp',3: 'Tail_lamp_left',4: 'Tail_light_left',5: 'Tail_lamp_right',6: 'Tail_light_right',7: 'Innova',8: 'Hycross',9: 'Toyota_logo',10: 'ZX',11: 'Hybrid',12: 'suzuki_logo',13: 'Invicto', 14: 'VX'}
    
    num_labels_map_dict_cmpv_combi = {
    0: 'Brake warning indicator (Red)',
    1: 'ECB warning indicator (Yellow)',
    2: 'ABS indicator',
    3: 'Break hold indicator',
    4: 'VSC off indicator',
    5: 'VSE indicator',
    6: 'Hold stand by indicator',
    7: 'EPS indicator',
    8: 'RR seat belt indicator',
    9: 'Ignition indicator',
    10: 'Seat belt indicator',
    11: 'Fuel level warning indicator',
    12: 'PKB Indicator',
    13: 'Air bag indicator',
    14: 'High beam indicator',
    15: 'Tail indicator',
    16: 'Turn indicator RH',
    17: 'Turn indicator LH',
    18: 'RR seat belt indicator_6',
    19: 'Eco mode indicator',
    20: 'Ready indicator',
    21: 'Ev mode indicator',
    22: 'Power mode indicator',
    23: 'Clearance sonar indicator',
    24: 'TMPS Indicator',
    25: 'LDA Indicator',
    26: 'LTA Indicator',
    27: 'PCS Off indicator',
    28: 'Inte tell lamp',
    29: 'Front Fog lamp indicator',
    30: 'AHB',
    31: 'ACC',
    32: 'Cruise'
}

    cls_name_number_map_dict = swap_keys_values(num_labels_map_dict_cmpv_combi)

    labels_count_dict = dict.fromkeys(cls_name_number_map_dict, 0)
    total_labels_count_dict = dict.fromkeys(cls_name_number_map_dict, 0)
     

    min_instances = 10  #choosen based on the smallest minority class

    # # # #count classes in xml
    total_labels_count_dict = read_xml_count_classes(xml_files_dir)
    print(total_labels_count_dict)
    plot_name = "combi_class_distribution_cmpv"
    
    plot_class_disribution(total_labels_count_dict, plot_name, plt_save_dir)

    # # # xml to yolo txt conversion fn
    # xml2yolo_txt(xml_files_dir, txt_output_dir, images_folder, cls_filename)
    
    total_cls_instance_count_dict = total_class_count_from_txt_folder(txt_output_dir)
    print("Total labels dict: ", total_cls_instance_count_dict)
    print("sorted keys :", sorted(total_cls_instance_count_dict.keys()))

    # # #count classes in generated txt
    # total_labels_count_dict = total_class_count_from_txt(txt_output_dir, total_labels_count_dict, num_labels_map_dict_highcross)   #reading class count from txt yolo format files
    # print("Total labels dict: ", total_labels_count_dict)




    # ********************************************************************************
    """
    balance_classes_by_masking : use for BASE MODEL creation, gets all classes initially, when enough for any class reached, then start masking only those classes
    mask_allclasses_except1_sampling : use for FINETUNING, use when only specific class is required for finetuning model
    """
    req_count = 200
    img_folder_path = images_folder
    txt_output_dir = txt_output_dir
    op_save_path = 'd:\Combi_annotations/balanced_dataset_200'
    mask_type = 'interpolate'   # other types: 'black', 'interpolate'
    req_class_list = list(cls_name_number_map_dict.keys())
    
    if not os.path.isdir(op_save_path):
        os.mkdir(op_save_path)
    
    balance_classes_by_masking(req_class_list, req_count, img_folder_path, txt_output_dir, num_labels_map_dict_cmpv_combi, cls_name_number_map_dict, op_save_path, mask_type)
    
    # ****************************************************************************
    
    
     # ******************************************************************
    ## USE BELOW SCRIPTS FOR FETCHING SPECIFIC CLASSES FOR FINETUNING THE MODEL
    
    ##filtering txt files, data to include minority classes and get minimum dataset
    # overall_minority_cls_list = ['glow_plug_warning_light', 'rear_fog_Lamp_Indicator', 'rr_diff_lock', 'fort_eco_mode_indicator_light', '4_lo_mode_indication', 'eco_run_on', 'imt_green']
    # labels_count_dict = minority_class_sample_and_move(images_folder, txt_files_dir, num_labels_map_dict, overall_minority_cls_list, labels_count_dict, min_instances, filtered_min_data_folder)
    # print("count dict:", labels_count_dict)

    ## list of classes to sample images for, use this function while finetuning model and adding weaker classes
    # # # BELOW CODE TO CREATE COMBI VAL SET WITH MINIMUM INSTANCES AND THEN REST AS TRAINING SET
    # highcross_val_list = ['left_turn_signal_indicator_light', 'airbag_allert', 'sunroof', 'check_engine_warning_light', 'brake_warning_Light_without_ABS', 'Brake_hold_standby', 'tail_lamp_position_indicator_light']
    # labels_count_dict, copied_img_list = fetch_specific_class(images_folder, txt_output_dir, num_labels_map_dict_highcross, highcross_val_list, labels_count_dict, min_instances, filtered_min_data_folder)
    # print("Filtered images labels :", labels_count_dict)

    # # fetch_cls_not_in_list(images_folder, txt_output_dir, copied_img_list, "D:\highcross\highcross_data\combi_annotated/train_v1")

    # **************************************************************************

    
    
    
    # ****************** Multiple masking techniques below******************************
    
    # # # MASKING ALL CLASSES EXCEPT 1 to BALANCE DATASET 
    # # req_class_list = ['highway_drive', 'right_turn_signal_indicator_light']
    # req_class_list = ['seat_belt_warning_light', 'air_bag_warning_light', 'fuel_Level_warning_light', 'check_engine_warning_light', 'power_off_light', 'vfc_indicator', 'Brake_hold', 'abs_indicator_light', 'slip_vsc_indicator_light', 'pkb_indicator_light', 'Brake_hold_standby', 'brake_warning_Light_without_ABS', 'brake_warning_Light_with_ABS', 'rear_seat', 'sunroof', 'ABS_off', 'tail_lamp_position_indicator_light', 'high_beam_indicator_light', 'slip_vsc_indicator_light_on', 'airbag_allert', 'accident_alert', 'highway_drive', 'road_view', 'ready_to_drive', 'right_turn_signal_indicator_light', 'left_turn_signal_indicator_light', 'fuel_Level_warning_light_2', 'ev_indicator', 'front_fog_lamp_indicator_light', 'rr_diff_lock', 'lock','eco_indicator_1', 'eco_indicator_2', 'sunroof', 'power_off_light', 'Temperature_warning', 'Drive_start_control']
    # min_instances = 50
    # img_folder_path = 'D:\highcross_newdata\combi_train_data\AB_Annotated'
    # label_file_folder = 'D:\highcross_newdata\combi_train_data\AB_Annotated/txt'
    # op_save_path = 'D:\highcross_newdata\combi_train_data\AB_Annotated/balanced_masked_data'
    
    # cls_name_number_map_dict = cls_name_number_map_dict

    # if not os.path.isdir(op_save_path):
    #     os.mkdir(op_save_path)
    # mask_allclasses_except1_sampling(req_class_list, min_instances, img_folder_path, label_file_folder, num_labels_map_dict_highcross, cls_name_number_map_dict, op_save_path)

    
    ### TESTING DATA ANNOTATION HAS NOT MISSED ANY CLASS LABELLING
    # img_folder_path = 'D:\highcross_newdata\combi_train_data\AX_Annotated'
    # txt_output_dir = 'D:\highcross_newdata\combi_train_data\AX_Annotated/txts'
    # op_save_path = 'D:\highcross_newdata\combi_train_data\AX_Annotated/all_masked'
    # mask_all_test_annotations(img_folder_path, txt_output_dir, num_labels_map_dict_highcross,  op_save_path)
