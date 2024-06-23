# yolo.py: This script is designed to load the YOLOv3 model and perform object detection on either images or videos based on user input through command-line arguments.



import numpy as np
import cv2 as cv
from yoloDetection import detectObject, displayImage
import sys

global class_labels, cnn_model, cnn_layer_names

def loadLibraries():
    global class_labels, cnn_model, cnn_layer_names
    
    # Load YOLOv3 model weights and class labels
    class_labels = open('yolov3model/yolov3-labels').read().strip().split('\n')
    print(f"Class labels: {class_labels}, Total: {len(class_labels)}")
    
    cnn_model = cv.dnn.readNetFromDarknet('yolov3model/yolov3.cfg', 'yolov3model/yolov3.weights')
    cnn_layer_names = cnn_model.getLayerNames()
    cnn_layer_names = [cnn_layer_names[i[0] - 1] for i in cnn_model.getUnconnectedOutLayers()]

def detectFromImage(imagename):
    # Function to detect objects from images
    label_colors = (0, 255, 0)  # Random colors to assign unique color to each label
    
    try:
        image = cv.imread(imagename)  # Read image
        image_height, image_width = image.shape[:2]  # Get image dimensions
    except:
        raise Exception('Invalid image path')
    finally:
        image, _, _, _, _ = detectObject(cnn_model, cnn_layer_names, image_height, image_width, image, label_colors, class_labels, 0)
        displayImage(image, 0)  # Display image with detected objects

def detectFromVideo(videoFile):
    # Function to detect objects from video
    label_colors = (0, 255, 0)  # Random colors to assign unique color to each label
    indexno = 0
    
    try:
        video = cv.VideoCapture(videoFile)  # Open video file
        frame_height, frame_width = None, None
        # Get video dimensions
        video_writer = None
    except:
        raise Exception('Unable to load video')
    finally:
        while True:
            frame_grabbed, frames = video.read()  # Read each frame from video
            
            if not frame_grabbed:  # Check if video loaded successfully
                break
            
            if frame_width is None or frame_height is None:
                frame_height, frame_width = frames.shape[:2]  # Get frame dimensions
            
            frames, _, _, _, _ = detectObject(cnn_model, cnn_layer_names, frame_height, frame_width, frames, label_colors, class_labels, indexno)
            # Display image with detected objects
            displayImage(frames, indexno)
            indexno += 1
            
            print(indexno)
        
        video.release()
        print("Releasing resources")

if __name__ == '__main__':
    loadLibraries()
    print("Sample commands to run code with image or video:")
    print("python yolo.py image input_image_path")
    print("python yolo.py video input_video_path")
    
    if len(sys.argv) == 3:
        if sys.argv[1] == 'image':
            detectFromImage(sys.argv[2])
        elif sys.argv[1] == 'video':
            detectFromVideo(sys.argv[2])
        else:
            print("Invalid input")
    else:
        print("Follow sample command to run code")
