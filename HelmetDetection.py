# HelmetDetection.py: This module uses Tkinter for a GUI application that integrates object detection using the YOLO model (yoloDetection.py). It allows uploading images, detecting objects (bikes and persons), and specifically detects helmets using another neural network model.


import numpy as np
import cv2 as cv
from yoloDetection import detectObject, displayImage
import sys
from tkinter import *
from tkinter import filedialog, messagebox
import pytesseract as tess
from keras.models import model_from_json

main = Tk()
main.title("Helmet Detection")
main.geometry("800x700")

global filename, class_labels, cnn_model, cnn_layer_names
global frame_count

frame_count = 0

# Loading models and labels
labels_value = []
with open("Models/labels.txt", "r") as file:
    for line in file:
        line = line.strip('\n')
        line = line.strip()
        labels_value.append(line)
    file.close()

with open('Models/model.json', "r") as json_file:
    loaded_model_json = json_file.read()

plate_detector = model_from_json(loaded_model_json)
plate_detector.load_weights("Models/model_weights.h5")
plate_detector._make_predict_function()

classesFile = "Models/obj.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfiguration = "Models/yolov3-obj.cfg"
modelWeights = "Models/yolov3-obj_2400.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def loadLibraries():
    global class_labels, cnn_model, cnn_layer_names
    
    # Load YOLOv3 model weights and class labels
    class_labels = open('yolov3model/yolov3-labels').read().strip().split('\n')
    print(f"Class labels: {class_labels}, Total: {len(class_labels)}")
    
    cnn_model = cv.dnn.readNetFromDarknet('yolov3model/yolov3.cfg', 'yolov3model/yolov3.weights')
    cnn_layer_names = cnn_model.getLayerNames()
    cnn_layer_names = [cnn_layer_names[i[0] - 1] for i in cnn_model.getUnconnectedOutLayers()]

def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="bikes")
    print("Image file loaded")

def detectBike():
    global option
    option = 0
    indexno = 0
    label_colors = (0,255,0)
    
    try:
        image = cv.imread(filename)
        image_height, image_width = image.shape[:2]
    except:
        raise Exception('Invalid image path')
    finally:
        image, ops = detectObject(cnn_model, cnn_layer_names, image_height, image_width, image, label_colors, class_labels, indexno)
        
        if ops == 1:
            displayImage(image, 0)  # Display image with detected objects
            option = 1
        else:
            displayImage(image, 0)

def drawPred(classId, conf, left, top, right, bottom, frame, option):
    global frame_count
    label = '%.2f' % conf
    
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
        
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    label_name, label_conf = label.split(':')
    print(label_name + " === " + str(conf) + "== " + str(option))
    
    if label_name == 'Helmet' and conf > 0.50:
        if option == 0 and conf > 0.90:
            cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
            frame_count += 1
        if option == 0 and conf < 0.90:
            cv.putText(frame, "Helmet Not detected", (10, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            frame_count += 1
    
    img = cv.imread(filename)
    img = cv.resize(img, (64, 64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 64, 64, 3)
    X = np.asarray(im2arr)
    X = X.astype('float32')
    X = X / 255
    preds = plate_detector.predict(X)
    predict = np.argmax(preds)
    
    textarea.insert(END, filename + "\n\n")
    textarea.insert(END, "Number plate detected as " + str(labels_value[predict]))

def postprocess(frame, outs, option):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    global frame_count_out
    frame_count_out = 0
    classIds = []
    confidences = []
    boxes = []
    classIds = []
    confidences = []
    boxes = []
    cc = 0

    # Remaining implementation of postprocess function is required

def videoHelmetDetect():
    global filename
    videofile = filedialog.askopenfilename(initialdir = "videos")
    cv.VideoCapture(videofile)
    while(True):
        ret, frame = video.read()
        if ret == True:
            frame_count = 0
            filename = "temp.png"
            cv.imwrite("temp.png",frame)
            cv.imshow('img',frame)
            if cv.waitKey(5) & 0xFF == ord('q'):
                break
        else:
            break
    video.release()
    cv.destroyAllWindows()

def detectHelmet():
    textarea.delete('1.0', END)
    global option
    if option == 1:
        frame_count = 0
        frame = cv.imread(filename)
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        postprocess(frame, outs, 1)
        t, _ = net.getPerfProfile()
        # label=''
        # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv.imshow("Predicted Result", frame)
        if cv.waitKey(0) & 0xFF == ord('q'):
            pass
    else:
        messagebox.showinfo("Person & Motor bike not detected in uploaded image", "Person & Motor bike not detected in uploaded image")

def exit():
    global main
    if cv.waitKey(5) & 0xFF == ord('q'):
        main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Number Plate Detection without Helmet', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=100, y=5)
title.pack()

font1 = ('times', 14, 'bold')
model = Button(main, text="Upload Image", command=upload)
model.place(x=200, y=100)
model.config(font=font1)

uploadimage = Button(main, text="Detect Motor Bike & Person", command=detectBike)
uploadimage.place(x=200, y=150)
uploadimage.config(font=font1)

classifyimage = Button(main, text="Detect Helmet", command=detectHelmet)
classifyimage.place(x=200, y=200)
classifyimage.config(font=font1)

exitapp = Button(main, text="Exit", command=exit)
exitapp.place(x=200, y=250)
exitapp.config(font=font1)

font1 = ('times', 12, 'bold')
textarea = Text(main, height=15, width=60)
scroll = Scrollbar(textarea)
textarea.configure(yscrollcommand=scroll.set)
textarea.place(x=10, y=300)
textarea.config(font=font1)

loadLibraries()
main.config(bg='light coral')
main.mainloop()
