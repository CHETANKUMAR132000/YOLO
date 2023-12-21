import xml.etree.ElementTree as ET
import glob
import os
import json
import re
    
cls_filename = '/media/harsha/New Volume/Harsha_Projects/CMPV_Data/front_cmpv.pbtxt'
lut = {} 
def read_combi_classes(combi_file_path):
    items = []
    
    with open(combi_file_path, 'r') as file:
        content = file.read()
        
        # Extract item blocks using regular expression
        item_blocks = re.findall(r'item {[\s\S]*?}', content)
        
        for item_block in item_blocks:
            item = {}
            
            # Extract id and name using regular expression
            id_match = re.search(r'id: (\d+)', item_block)
            name_match = re.search(r"name: '(.+)'", item_block)
            
            if id_match and name_match:
                
                item['id'] = int(id_match.group(1))
                item['name'] = name_match.group(1)
                lut[item['name']] = item['id'] -1   #in class file , num start from 1, yolo requires from 0
                if item['name'] not in items:
                    items.append(item['name'])
    print(lut)
    return lut

def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
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

def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


classes = []
# input_dir = "D:/Harsha/Combi_Data/xmls_50635/xmls_50635/"
input_dir = "/media/harsha/New Volume/Harsha_Projects/logo_error_Data/logo_error_classes_data/frames_sampled"
# output_dir = "D:/Harsha/Combi_Data/xmls_50635/labels_bbox_correct_classes"
output_dir = "/media/harsha/New Volume/Harsha_Projects/logo_error_Data/logo_error_classes_data/frames_sampled"

image_dir = "/media/harsha/New Volume/Harsha_Projects/logo_error_Data/logo_error_classes_data/frames_sampled"

# lut = read_combi_classes(cls_filename)
lut['logo'] = 0
# lut['fort_eco_mode_indicator_light'] = 34

# create the labels folder (output directory)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# identify all the xml files in the annotations folder (input directory)
files = glob.glob(os.path.join(input_dir, '*.xml'))
# loop through each 
count = 0
for fil in files:
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]
    # check if the label contains the corresponding image file
    # if not os.path.exists(os.path.join(image_dir, f"{filename}.jpg")):
    #     print(f"{filename} image does not exist!")
    #     continue

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

            
            # if label in lut:
            label_str = str(lut[label]) 
            # label_str = str(0) 
            pil_bbox = [int(x.text) for x in obj.find("bndbox")]
            yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
            # yolo_segm = xml_to_yolo_segment(pil_bbox, width, height)
            # convert data to string
            bbox_string = " ".join([str(x) for x in yolo_bbox])
            result.append(f"{label_str} {bbox_string}")
            # del root, tree
            # else:
            #     print("class label/name is not in list: ", label)
            # del root, tree


        if result:
            # generate a YOLO format text file for each xml file
            with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result))
        else:
            count += 1
    except Exception as e:
        # Handling empty XML files and printing the exact error message
        print(f"Error parsing XML : {str(e)}")
        count += 1
        # del root, tree

print("num of images skipped: ", count)

# generate the classes file as reference
with open('classes.txt', 'w', encoding='utf8') as f:
    f.write(json.dumps(classes))