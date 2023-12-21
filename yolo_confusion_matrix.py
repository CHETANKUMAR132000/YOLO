import csv
from ultralytics import YOLO
import multiprocessing


def metrics_process():
    # metrics.box.map    # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps   # a list contains map50-95 of each category
    # print(metrics.box.results_dict)
    # print(metrics.box.ap_class_index.results_dict)
    # print(metrics.box.maps)

    
    # Assuming you have a 'Metric' object named 'metric'
    # ap_scores = metrics.ap50()

    # # Print the AP scores for each class
    # for i, ap in enumerate(ap_scores):
    #     class_index = metric.ap_class_index[i]
    #     print(f"AP for class {class_index}: {ap}")
    pass

def main():
    model = YOLO("D:\Harsha\scripts/runs\detect\combi_dashboard1_90%/weights/best.pt", 'detect')
    # metrics = model.val(data="D:/Harsha/Combi_Data/all_data/dataset.yaml", conf=0.7, line_width=2, save=True, save_conf=True)
    metrics = model.val(data="D:\highcross_newdata\combi_model_data\dashboard_1_valdata/dataset.yaml", conf=0.7, save=True, save_conf=True)

    conf_matrix_output_csv = 'D:\highcross_newdata/combi_dash1_matrix.csv'

    # print(metrics.box.maps)   # a list contains map50-95 of each category
    # print("Printing conf matrix: ", metrics.confusion_matrix.matrix.shape)
    
    # header = ['master_caution_warning_light', 'brake_warning_Light', 'pkb_indicator_light', 'seat_belt_warning_light', 'air_bag_warning_light', 'urea_scr_adblue_warning_light', 'check_engine_warning_light', 'slip_vsc_indicator_light', 'slip_off_vsc_off_indicator', 'abs_indicator_light', 'fuel_Level_warning_light', 'glow_plug_warning_light', 'rear_fog_Lamp_Indicator', 'cruise_control_indicator_light_Green', 'high_beam_indicator_light', 'tail_lamp_position_indicator_light', 'front_fog_lamp_indicator_light', 'left_turn_signal_indicator_light', 'right_turn_signal_indicator_light', 'power_mode_indicator_light', 'eco_mode_indicator_light', 'eco_indicator_light', 'rr_diff_lock', '4_lo_mode_indication', 'vfc_indicator', 'auto_lsd_indicator_light', 'dac_system_light_indicator', '4wd_system_indication', 'imt_green', 'eco_run_off', 'eco_run_on', 'DPF', 'sport_mode_indicator_light', 'fort_power_mode_indicator_light', 'fort_eco_mode_indicator_light']
    header = ['seat_belt_warning_light', 'air_bag_warning_light', 'fuel_Level_warning_light', 'check_engine_warning_light', 'power_off_light', 'vfc_indicator', 'Brake_hold', 'abs_indicator_light', 'slip_vsc_indicator_light', 'pkb_indicator_light', 'Brake_hold_standby', 'brake_warning_Light_without_ABS', 'brake_warning_Light_with_ABS', 'rear_seat', 'sunroof', 'ABS_off', 'tail_lamp_position_indicator_light', 'high_beam_indicator_light', 'slip_vsc_indicator_light_on', 'airbag_allert', 'accident_alert', 'highway_drive', 'road_view', 'ready_to_drive', 'right_turn_signal_indicator_light', 'left_turn_signal_indicator_light', 'fuel_Level_warning_light_2', 'ev_indicator', 'front_fog_lamp_indicator_light', 'rr_diff_lock', 'lock','eco_indicator_1', 'eco_indicator_2', 'sunroof', 'power_off_light', 'Temperature_warning', 'Drive_start_control']
    header.insert(0, 'Class_Names')

    # Write the matrix to the CSV file
    with open(conf_matrix_output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header row
        writer.writerow(header)
        
        # Write the matrix rows
        for i, row in enumerate(metrics.confusion_matrix.matrix):
            writer.writerow([header[i]] + list(row))



if __name__ == '__main__':
    # Add freeze_support() to handle multiprocessing in Windows
    multiprocessing.freeze_support()
    
    main()
