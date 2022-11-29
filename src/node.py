#!/usr/bin/env python3

import rospy as rp
from model_detect.srv import *
from geometry_msgs.msg import Point
from detect import detect_block
import cv2

def getObjectPts(req):
    
    responseObjPts = Model_srvResponse()
    
    Positions = detect_block()
    
    for Position in Positions :
        
        tempPosition = Point()
        tempPosition.x = Position[0]
        tempPosition.y = Position[1]
        
        responseObjPts.block_at_frame.append(tempPosition)
        
    
    return responseObjPts
    

def main() : 
    rp.init_node("camera_service_node")
    
    rp.Service("/Model_srv", Model_srv, getObjectPts)
    
    rp.loginfo("Ready to receive request")
    
    # detect_block()
    
    rp.spin()

if __name__ == '__main__':
    main()
    # getObjectPts(0)