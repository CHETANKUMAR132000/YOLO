import os
import json
from tqdm import tqdm
import shutil

def make_folders(path="output"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path

def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    Convert bounding box from COCO  format to YOLO format

    Parameters
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[int]
        bounding box annotation in COCO format: 
        [top left x position, top left y position, width, height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format: 
        [x_center_rel, y_center_rel, width_rel, height_rel]
    """
    
    # YOLO bounding box format: [x_center, y_center, width, height]
    # (float values relative to width and height of image)
    x_tl, y_tl, w, h = bbox

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    x = x_center * dw
    y = y_center * dh
    w = w * dw
    h = h * dh

    return [x, y, w, h]

coco_valid_classes = {
    1: 0,   # Person
    2: 1,   # Bicycle
    3: 2,   # Car
    4: 3,   # Motorcycle
    5: 4,   # Airplane
    6: 5,   # Bus
    7: 6,   # Train
    8: 7,   # Truck
    9: 8,   # Boat
    10: 9,  # Traffic light
    11: 10, # Fire hydrant
    13: 11, # Stop sign
    14: 12, # Parking meter
    15: 13, # Bench
    16: 14, # Bird
    17: 15, # Cat
    18: 16, # Dog
    19: 17, # Horse
    20: 18, # Sheep
    21: 19, # Cow
    22: 20, # Elephant
    23: 21, # Bear
    24: 22, # Zebra
    25: 23, # Giraffe
    27: 24, # Backpack
    28: 25, # Umbrella
    31: 26, # Handbag
    32: 27, # Tie
    33: 28, # Suitcase
    34: 29, # Frisbee
    35: 30, # Skis
    36: 31, # Snowboard
    37: 32, # Sports ball
    38: 33, # Kite
    39: 34, # Baseball bat
    40: 35, # Baseball glove
    41: 36, # Skateboard
    42: 37, # Surfboard
    43: 38, # Tennis racket
    44: 39, # Bottle
    46: 40, # Wine glass
    47: 41, # Cup
    48: 42, # Fork
    49: 43, # Knife
    50: 44, # Spoon
    51: 45, # Bowl
    52: 46, # Banana
    53: 47, # Apple
    54: 48, # Sandwich
    55: 49, # Orange
    56: 50, # Broccoli
    57: 51, # Carrot
    58: 52, # Hot dog
    59: 53, # Pizza
    60: 54, # Donut
    61: 55, # Cake
    62: 56, # Chair
    63: 57, # Couch
    64: 58, # Potted plant
    65: 59, # Bed
    67: 60, # Dining table
    70: 61, # Toilet
    72: 62, # TV
    73: 63, # Laptop
    74: 64, # Mouse
    75: 65, # Remote
    76: 66, # Keyboard
    77: 67, # Cell phone
    78: 68, # Microwave
    79: 69, # Oven
    80: 70, # Toaster
    81: 71, # Sink
    82: 72, # Refrigerator
    84: 73, # Book
    85: 74, # Clock
    86: 75, # Vase
    87: 76, # Scissors
    88: 77, # Teddy bear
    89: 78, # Hair drier
    90: 79, # Toothbrush
}

def convert_coco_json_to_yolo_txt(output_path, json_file):

    path = make_folders(output_path)

    with open(json_file) as f:
        json_data = json.load(f)

    # Write _darknet.labels file with valid COCO classes
    label_file = os.path.join(output_path, "_darknet.labels")
    with open(label_file, "w") as f:
        for coco_id, valid_id in coco_valid_classes.items():
            category = next(cat for cat in json_data["categories"] if cat["id"] == coco_id)
            category_name = category["name"]
            f.write(f"{category_name}\n")

    for image in tqdm(json_data["images"], desc="Annotation txt for each image"):
        img_id = image["id"]
        img_name = image["file_name"]
        img_width = image["width"]
        img_height = image["height"]

        anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
        anno_txt = os.path.join(output_path, img_name.split(".")[0] + ".txt")
        with open(anno_txt, "w") as f:
            for anno in anno_in_image:
                category = anno["category_id"]
                if category in coco_valid_classes:
                    bbox_coco = anno["bbox"]
                    x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_coco)
                    valid_class_id = coco_valid_classes[category]
                    f.write(f"{valid_class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


if __name__ == "__main__":
    convert_coco_json_to_yolo_txt("D:\latest\datasets\coco_val2017/dataset_valid", "D:\latest\datasets\coco_val2017/instances_val2017.json")