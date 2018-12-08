import numpy as np
import cv2 as cv2
# cv2.ocl.setUseOpenCL(False)  #try to use if there are problems

# print(cv2.__version__)
#### lots of the code is from these tutorials:
## for SIFT
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html
## for feature mapping
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#
## for object detection
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#
###############################################################################################################
# Verbose = True
Verbose = False

def extract_features(img):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create(sigma=2, nfeatures=30,
                    nOctaveLayers=3,contrastThreshold=.00)
    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(img,None)
    # return keypoints and descriptors
    return (kp, des)

def find_matches(des1,des2):
    # find matches
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    if len(des1) <= 3 or len(des2) <= 3: return None
    matches = flann.knnMatch(des1,des2,k=2)
    return matches


def filter_matches(matches,threshold=0.7):
    # find all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance > threshold*n.distance:
            good.append([m,n])
    return good


def create_mask(good,kp1,kp2):
    # creates a mask
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    return (matchesMask, M)


def create_display_image(img1,img2,kp1,kp2,matches,good,matchesMask,M,MIN_MATCH_COUNT=10,NUM_DISPLAY_MATCHES="good"):
    # convert img2 for display
    h,w,d = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    # find best matches (for display)
    if NUM_DISPLAY_MATCHES == "good":
        display_matches=good
    else:
        #cap the NUM_DISPLAY_MATCHES to the number of matches
        NUM_DISPLAY_MATCHES=min(NUM_DISPLAY_MATCHES,len(matches))
        display_matches = sorted(matches, key = lambda x:x[1].distance-x[0].distance,reverse=True)
        display_matches = display_matches[:NUM_DISPLAY_MATCHES]

    #create output image
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,display_matches,None,flags=2)
    return img3


def get_matches_from_details(kp1, des1, kp2, des2, MIN_MATCH_COUNT=10,mask=True):
    # find matches (uses Knn)
    matches = find_matches(des1,des2)
    if matches is None: return (None, None, None, None)

    # find all the good matches as per Lowe's ratio test.
    good = filter_matches(matches,threshold=.7)
    if good is None: return(None,None,None,None)

    if len(good)>=MIN_MATCH_COUNT and mask is True:
        # create a mask
        matchesMask, M = create_mask(good,kp1,kp2)
        return (matches, good, matchesMask, M)
    else:
        if mask is True and Verbose is True: print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        return (None,None,None,None)

def compare_images(img1,img2,MIN_MATCH_COUNT=10,NUM_DISPLAY_MATCHES="good"):
    img1 = img1.astype('uint8')
    img2 = img2.astype('uint8')
    # resize, needed for testing or if the images have diffferent resolution
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1],img1.shape[0]), interpolation=cv2.INTER_CUBIC)
    # get features
    kp1, des1 = extract_features(img1)
    kp2, des2 = extract_features(img2)
    if des1 is None or des2 is None or len(des1) <= 3 or len(des2) <= 3: return None
    # get matches
    matches, good, matchesMask, M = get_matches_from_details(kp1, des1, kp2, des2,
                                                    MIN_MATCH_COUNT=MIN_MATCH_COUNT,mask=True)
    if M is None or good is None: return None
    #create output image
    img3 = create_display_image(img1,img2,kp1,kp2,matches,good,matchesMask,M,
                    MIN_MATCH_COUNT=MIN_MATCH_COUNT, NUM_DISPLAY_MATCHES=NUM_DISPLAY_MATCHES)
    if img3 is None: return None
    return (img3,True)

if __name__ == '__main__':
    img1 = cv2.imread("sift_samples/test_pic.jpg") # queryImage
    img1 = img1[100:400,100:400,:] #crop image1

    img2 = cv2.imread("sift_samples/test_pic.jpg") # testImage
    img2 = np.flip(np.swapaxes(img2,0,1), 0) # filped

    img3,_ = compare_images(img1,img2,NUM_DISPLAY_MATCHES=5)
    cv2.imwrite("sift_samples/matching_test2.jpg",img3)
    print("done")
