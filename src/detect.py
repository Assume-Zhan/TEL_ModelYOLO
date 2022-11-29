import cv2 as cv
import os
import time

def detect_block() : 
    this_dir = os.path.dirname(__file__) 
    filename = os.path.dirname(os.path.realpath(__file__))
    
    Conf_threshold = 0.8
    NMS_threshold = 0.4
    COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
            (255, 255, 0), (255, 0, 255)]

    class_name = []
    with open(filename + '/data/block.names', 'r') as f: # change to your own path
        class_name = [cname.strip() for cname in f.readlines()]
    print(class_name)

    net = cv.dnn.readNet(filename + '/weight/yolov4-tiny-custom_best.weights', filename + '/cfg/yolov4-tiny-custom.cfg') 
    
    model = cv.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

    cam = cv.VideoCapture(0)

    check, img = cam.read()

    classes, scores, boxes = model.detect(img, Conf_threshold, NMS_threshold)
    
    tel_position = []

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_name[classid], score)
        
        if class_name[classid] == 'C' or class_name[classid] == 'F' :
            continue
        
        tel_position.append([box[0] + box[2] / 2, box[1] + box[3] / 2])
        
        print(label, box) # (x, y, width, height)
        
    
    return tel_position
    
if __name__ == "__main__" :
    detect_block()