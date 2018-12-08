#!/usr/bin/env python
###################

from __future__ import print_function
import roslib
# roslib.load_manifest('cv2ros2')
import rospy
import sys
import cv2
import numpy as np
from matching_script import *
from std_msgs.msg import String #for indigo up
from sensor_msgs.msg import Image #sensor read
from cv_bridge import CvBridge, CvBridgeError #CvBridge
import PIL

### code for filtering noise
## not fully implemented
# def background_filter_mask(input_image, comparison_image, comparison_mask):
#     global counter
#     counter+=1
#     if counter % 100 is 0: comparison_mask = (comparison_mask1+1)
#     difference_image = np.abs(comparison_image-input_image)
#     differences= np.where(difference_image > 150)
#     comparison_mask[differences] = comparison_mask[differences] * 100-5
#     next_mask = (comparison_mask[:,:,0] + comparison_mask[:,:,1] + comparison_mask[:,:,2])/3
#     for i in [0,1,2]: comparison_mask[:,:,i]=next_mask
#     return (input_image, comparison_mask)
# comparison_image = np.copy(first_image)
# comparison_mask = np.zeros_like(first_image)+1
# comparison_image, comparison_mask = background_filter_mask(input_image, comparison_image, comparison_mask)

def get_img(data):
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")#blue-green-red
    # cv_image = bridge.imgmsg_to_cv2(data, "rgb8")#red-green-blue
    return cv_image

##################### callback ############################
def callback(data):
    global run_once, first_image, bridge, comparison_image, comparison_mask, counter
    if run_once is False:
        run_once=True

        first_image = get_img(data)

    else:
        try:
            input_image = bridge.imgmsg_to_cv2(data, "bgr8") #blue-green-red
            # input_image = bridge.imgmsg_to_cv2(data, "rgb8") #red-green-blue 
        except CvBridgeError as e:
            print(e)

        # (rows,cols,channels) = input_image.shape
        # if cols > 60 and rows > 60 :
        #     cv2.circle(input_image, (200,200), 100, 255) #circle around PoI
        output_image,success = compare_images(first_image,input_image,MIN_MATCH_COUNT=10,NUM_DISPLAY_MATCHES=5)
        if output_image is None: output_image = input_image

        
        cv2.imshow("sift_node", output_image)

        cv2.waitKey(2)
        if success is True or counter%5 == 0:
            try:
                counter=0
                image_pub.publish(bridge.cv2_to_imgmsg(output_image, "bgr8")) #red-green-blue
                # image_pub.publish(bridge.cv2_to_imgmsg(output_image, "rgb8")) #red-green-blue
            except CvBridgeError as e:
                print(e)
        else: counter+=1

##################### main ############################
def main(args):
    global image_pub, run_once, bridge, counter
    bridge = CvBridge()
    run_once=False
    counter=0
    image_pub = rospy.Publisher("/sift/image_raw",Image, queue_size=2)
    rospy.init_node('sift_node', anonymous=True)
    try:
        image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,callback)
        rospy.spin()
    except KeyboardInterrupt:
        print("close")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
