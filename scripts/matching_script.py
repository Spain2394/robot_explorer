import numpy as np
import cv2 as cv2
cv2.ocl.setUseOpenCL(False)  #try to use if there are problems
print(cv2.__version__)

def extract_features(img):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
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
    matches = flann.knnMatch(des1,des2,k=2)
    return matches


def filter_matches(matches,threshold=0.7):
    # find all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
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
        display_matches = sorted(matches, key = lambda x:x[1].distance-x[0].distance)
        display_matches = display_matches[:NUM_DISPLAY_MATCHES]

    #create output image
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,display_matches,None,flags=2)
    return img3


def get_matches_from_details(kp1, des1, kp2, des2, MIN_MATCH_COUNT=10,mask=True):
    # find matches (uses Knn)
    matches = find_matches(des1,des2)

    # find all the good matches as per Lowe's ratio test.
    good = filter_matches(matches,threshold=0.7)

    if len(good)>MIN_MATCH_COUNT and mask is True:
        # create a mask
        matchesMask, M = create_mask(good,kp1,kp2)
    else:
        if mask is True: print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        M = None
    return (matches, good, matchesMask, M)


def compare_images(img1,img2,MIN_MATCH_COUNT=10,NUM_DISPLAY_MATCHES="good"):
    # resize, needed for testing or if the images have diffferent resolution
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1],img1.shape[0]), interpolation=cv2.INTER_CUBIC)
    # get features
    kp1, des1 = extract_features(img1)
    kp2, des2 = extract_features(img2)
    # get matches
    matches, good, matchesMask, M = get_matches_from_details(kp1, des1, kp2, des2,
                                                    MIN_MATCH_COUNT=MIN_MATCH_COUNT,mask=True)
    #create output image
    if not matchesMask is None:
        img3 = create_display_image(img1,img2,kp1,kp2,matches,good,matchesMask,M,
                    MIN_MATCH_COUNT=MIN_MATCH_COUNT, NUM_DISPLAY_MATCHES=NUM_DISPLAY_MATCHES)
    else:
        img3 = None
    return img3


if __name__ == '__main__':
    img1 = cv2.imread("test_pic.jpg") # queryImage
    img1 = img1[100:400,100:400,:] #crop image1

    img2 = cv2.imread("test_pic.jpg") # testImage
    img2 = np.flip(np.swapaxes(img2,0,1), 0) # flipped

    img3 = compare_images(img1,img2,NUM_DISPLAY_MATCHES=10)
    cv2.imwrite("matching_test2.jpg",img3)
