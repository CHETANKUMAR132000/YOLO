"""
    Author : Harsha
    Deevia Software Ltd
"""
import cv2
import os
import glob

# Function to extract the first 20 frames from a video
def extract_frames(video_path, output_folder):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      
    # Loop through the first 20 frames
    # for i in range(1):
        # Read the frame
    #read first frame
    for i in range(1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i) 
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Empty frame")
            # Save the frame as a PNG file
        else:
            frame_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_" + f"{i}" + ".png")    
            cv2.imwrite(frame_path, frame)
            
    #     # read middle frame and save
    # mid_frame_no = int(n_frames//2)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 29) #saving 19th frame as that has some light variation
    # ret, frame = cap.read()
    # if not ret or frame is None:
    #     print("Empty frame")
    #     # Save the frame as a PNG file
    # else:
    #     frame_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_" + f"{1}" + ".png")    
    #     cv2.imwrite(frame_path, frame)
        
    # Release the video capture object
    cap.release()

# Function to process all videos in a folder
def process_videos(folder_path, output_folder):
    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a video
        if file_name.endswith(".mp4") or file_name.endswith(".avi"):
            # if not file_name.split("_")[-1].split(".")[0] == "detection":
            video_path = os.path.join(folder_path, file_name)
            extract_frames(video_path, output_folder)

# Example usage
if __name__ == "__main__":
    #input_folders_path = "D:/FrontVideos"
    input_folders_path = "/media/harsha/New Volume/Harsha_Projects/logo_error_Data"
    output_folder_path = "/media/harsha/New Volume/Harsha_Projects/logo_error_Data/frames"
    # input_folders_path_list = os.listdir(input_folders_path)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    input_folders_path_list = glob.glob(input_folders_path, recursive=True)
    for folder_name in input_folders_path_list:
        folder_path = os.path.join(input_folders_path, folder_name  )
        print(folder_path)
        process_videos(folder_path, output_folder_path)
