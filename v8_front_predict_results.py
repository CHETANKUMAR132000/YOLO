from ultralytics import YOLO
import csv



def generate_prob_results(pred_dict, pred_prob):
    # count = 0
    results_csv_path = 'D:/Harsha/Annotated_Dataset/2Car_Models_Min_Data/v8_results.csv'
    if pred_dict == {}:
        #change the class names accordlingly
        pred_dict = {'front_fog_cover_left': 0,	'front_fog_cover_left_RG': 0,	'front_fog_cover_right': 0,	'front_fog_cover_right_RG':0,	'radiator_grill':0,   'radiator_grill_RG':0,	'headLamp_right':0,  'headLamp_right_RG':0,  'headLamp_left':0,  'headLamp_left_RG':0,	'front_turn_right':0,	'front_turn_left':0,	'front_logo':0,	'radiator_lower_grill':0,	'right_mirror':0,	'left_mirror':0}
        # pred_dict = {class_name : 0  for class_name in names.values()}
        
        pred_prob = {'front_fog_cover_left': [],	'front_fog_cover_left_RG': [],	'front_fog_cover_right': [],	'front_fog_cover_right_RG':[],	'radiator_grill':[],   'radiator_grill_RG':[],	'headLamp_right':[],  'headLamp_right_RG':[],  'headLamp_left':[],  'headLamp_left_RG':[],	'front_turn_right':[],	'front_turn_left':[],	'front_logo':[],	'radiator_lower_grill':[],	'right_mirror':[],	'left_mirror':[]}

    
    print("Predict dict: ", pred_dict)
    #print("Confidence dict:", pred_prob)
    for cls_name in pred_prob.keys():
        
        vals = pred_prob[cls_name]  
        if len(vals) !=  0:
            avg_prob = round(sum(vals)/len(vals), 3)
            max_conf = round(max(vals), 3)
            min_conf = round(min(vals), 3)
        else:
            avg_prob = 0
            max_conf = 0
            min_conf = 0
        pred_prob[cls_name] = {"avg": avg_prob, "max": max_conf, "min": min_conf }
        
    print("Confidence dict:", pred_prob)
    header = pred_dict.keys()
    accuracy = pred_dict.values()
    print("No of images in this dataset:", len(vals))  # this value needs to be redone
    percent_acc = [round(acc/len(vals), 3) for acc in list(accuracy)]  #  round(list(accuracy)/len(pred_prob[cls_name]), 3)
    mean_acc = [dict["avg"] for dict in pred_prob.values()]
    max_acc = [dict["max"] for dict in pred_prob.values()]
    min_acc = [dict["min"] for dict in pred_prob.values()]
    with open(results_csv_path, 'a') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(header)
        writer.writerow(accuracy)
        writer.writerow(percent_acc)
        writer.writerow(mean_acc)
        writer.writerow(max_acc)
        writer.writerow(min_acc)





# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("D:/Harsha/runs/segment/v8_2car_580epochs/weights/best.pt", 'segment')  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set

results = model.predict(source="D:/Harsha/Annotated_Dataset/2Car_Models_Min_Data/Large_TestSet/RG_Frames", conf=0.7, line_width=2, retina_masks=False, save=True, save_conf=True)  # predict on an image
car_model_name = "RG"
error_txt_path = 'D:/Harsha/Annotated_Dataset/2Car_Models_Min_Data/v8_errors.txt'

# print(results[0])
boxes = results[0].boxes
# print(boxes.conf, boxes.cls)  # confidence score, (N, ))

#confidence scores written to csv to check for non detections and confident(higher conf score) mis-classifications
pred_dict = {'front_fog_cover_left': 0,	'front_fog_cover_left_RG': 0,	'front_fog_cover_right': 0,	'front_fog_cover_right_RG':0,	'radiator_grill':0,   'radiator_grill_RG':0,	'headLamp_right':0,  'headLamp_right_RG':0,  'headLamp_left':0,  'headLamp_left_RG':0,	'front_turn_right':0,	'front_turn_left':0,	'front_logo':0,	'radiator_lower_grill':0,	'right_mirror':0,	'left_mirror':0}        
pred_prob = {'front_fog_cover_left': [],	'front_fog_cover_left_RG': [],	'front_fog_cover_right': [],	'front_fog_cover_right_RG':[],	'radiator_grill':[],   'radiator_grill_RG':[],	'headLamp_right':[],  'headLamp_right_RG':[],  'headLamp_left':[],  'headLamp_left_RG':[],	'front_turn_right':[],	'front_turn_left':[],	'front_logo':[],	'radiator_lower_grill':[],	'right_mirror':[],	'left_mirror':[]}
cls_mapping =  {0: 'front_fog_cover_left',1: 'front_turn_right', 2: 'front_logo', 3:'front_turn_left', 4: 'headLamp_left_RG', 5:'front_fog_cover_left_RG', 6: 'right_mirror',7:  'headLamp_right',8: 'headLamp_left',9: 'radiator_grill',10: 'radiator_lower_grill',11: 'radiator_grill_RG',12: 'headLamp_right_RG',13: 'front_fog_cover_right_RG',14: 'left_mirror',15: 'front_fog_cover_right'}
# results = results.numpy()
# results = results.tolist()
for result in results:
    # print(result)
    for lst_index, cls_num  in enumerate(result.boxes.cls.tolist()):
        class_name = cls_mapping[int(cls_num)]
        # print(class_name, cls_num)
        pred_dict[class_name] += 1
        # print(pred_dict)
        conf = result.boxes.conf.tolist()[lst_index]
        pred_prob[class_name].append(float(conf))
        with open(error_txt_path, 'a') as file:
            # writer = csv.writer(file, lineterminator='\n')

            if car_model_name == "RG":
                if class_name in ['front_fog_cover_left', 'front_fog_cover_right', 'radiator_grill', 'headLamp_right', 'headLamp_left']: #i.e misclassified
                    # print(result.path)
                    file.write(result.path)
                    file.write('\n')
                    

    # break

generate_prob_results(pred_dict, pred_prob)

# path = model.export(format="onnx")  # export the model to ONNX format