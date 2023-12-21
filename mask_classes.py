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
    labels_count_dict = {}
    num_labels_map_dict = {0: 'master_caution_warning_light', 1: 'brake_warning_Light',2: 'pkb_indicator_light',3: 'seat_belt_warning_light',4: 'air_bag_warning_light',5: 'urea_scr_adblue_warning_light',6: 'check_engine_warning_light',7: 'slip_vsc_indicator_light',8: 'slip_off_vsc_off_indicator',9: 'abs_indicator_light',10: 'fuel_Level_warning_light',11: 'glow_plug_warning_light',12: 'rear_fog_Lamp_Indicator',13: 'cruise_control_indicator_light_Green',14: 'high_beam_indicator_light',15: 'tail_lamp_position_indicator_light',16: 'front_fog_lamp_indicator_light',17: 'left_turn_signal_indicator_light',18: 'right_turn_signal_indicator_light',19: 'power_mode_indicator_light',20: 'eco_mode_indicator_light',21: 'eco_indicator_light',22: 'rr_diff_lock',23: '4_lo_mode_indication',24: 'vfc_indicator',25: 'auto_lsd_indicator_light',26: 'dac_system_light_indicator',27: '4wd_system_indication',28: 'imt_green',29: 'eco_run_off',30: 'eco_run_on',31: 'DPF',32: 'sport_mode_indicator_light',33: 'fort_power_mode_indicator_light', 34: 'fort_eco_mode_indicator_light'}
    num_labels_map_dict_highcross = {0: 'seat_belt_warning_light',1: 'air_bag_warning_light',2: 'fuel_Level_warning_light',3: 'check_engine_warning_light',4: 'power_off_light',5: 'vfc_indicator',6: 'Brake_hold',7: 'abs_indicator_light',8: 'slip_vsc_indicator_light',9: 'pkb_indicator_light',10: 'Brake_hold_standby',11: 'brake_warning_Light_without_ABS',12: 'brake_warning_Light_with_ABS',13: 'rear_seat',14: 'sunroof',15: 'ABS_off',16: 'tail_lamp_position_indicator_light',17: 'high_beam_indicator_light',18: 'slip_vsc_indicator_light_on',19: 'airbag_allert',20: 'accident_alert',21: 'highway_drive',22: 'road_view',23: 'ready_to_drive',24: 'right_turn_signal_indicator_light',25: 'left_turn_signal_indicator_light', 26: 'fuel_Level_warning_light_2', 27: 'ev_indicator', 28: 'front_fog_lamp_indicator_light', 29: 'rr_diff_lock', 30: 'lock', 31: 'eco_indicator_1', 32: 'eco_indicator_2', 33: 'sunroof',34: 'power_off_light',35: 'Temperature_warning', 36: 'Drive_start_control', 37: 'beam'}
    num_labels_map_dict_highcross_rear = {0:'Sharkfin_Antenna',1: 'Rear_Spoiler',2: 'High_mount_stop_lamp',3: 'Tail_lamp_left',4: 'Tail_light_left',5: 'Tail_lamp_right',6: 'Tail_light_right',7: 'Innova',8: 'Hycross',9: 'Toyota_logo',10: 'ZX',11: 'Hybrid',12: 'suzuki_logo',13: 'Invicto', 14: 'VX'}

    xml2yolo_txt(xml_files_dir, txt_output_dir, images_folder, cls_filename)
    
    ### TESTING DATA ANNOTATION HAS NOT MISSED ANY CLASS LABELLING
    img_folder_path = 'D:\highcross_newdata\combi_train_data\AG_Annotated'
    txt_output_dir = 'D:\highcross_newdata\combi_train_data\AG_Annotated/txts'
    op_save_path = 'D:\highcross_newdata\combi_train_data\AG_Annotated/all_masked'
    mask_all_test_annotations(img_folder_path, txt_output_dir, num_labels_map_dict_highcross,  op_save_path)
