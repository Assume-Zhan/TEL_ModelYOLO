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
    img = cv.imread(filename + '/example.png') # Put the Picture that you want to detect blocks

    classes, scores, boxes = model.detect(img, Conf_threshold, NMS_threshold)

    print(img.shape[0] , img.shape[1])
    
    tel_position = []

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_name[classid], score)
        
        if class_name[classid] == 'C' or class_name[classid] == 'F' :
            continue
        
        tel_position.append([box[0], box[1]])
        
        print(label, box) # (x, y, width, height)
        pred_img = cv.rectangle(img, box, color, 1)
        pred_img = cv.putText(pred_img, label, (box[0], box[1]-10), cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
    
    return tel_position
    
if __name__ == "__main__" :
    detect_block()