import os
from ultralytics import YOLO
import multiprocessing

def main():
    base_path = "D:/highcross_newdata/COMBI"
    folders = os.listdir(base_path)
    print(folders)
    for folder_ in folders:
        folder_path = os.path.join(base_path, folder_)
        # print(folder_path)
        # yolo detect predict model="D:/Harsha/scripts/runs/detect/rear_base_model/weights/best.pt"  source="D:/highcross_newdata/REAR" conf=0.7 line_width=2
        model = YOLO("D:/Harsha/scripts/runs/detect/balanced_40_interpolate_mask/weights/best.pt", 'detect')
        # metrics = model.val(data="D:/Harsha/Combi_Data/all_data/dataset.yaml", conf=0.7, line_width=2, save=True, save_conf=True)
        model.predict(source=folder_path, conf=0.7, save=True, save_conf=True)




if __name__ == '__main__':
    # Add freeze_support() to handle multiprocessing in Windows
    multiprocessing.freeze_support()
    
    main()

