#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
from cv2 import LINE_AA
import numpy as np
import matplotlib.pyplot as plt

def trim(frame):
    # convert to gray scale
    frame_g = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    height,width = frame_g.shape 

    # calculate bounding box
    height_box = 0
    width_box = 0
    got_start_y = False
    got_start_x = False
    start_x = 0
    start_y = 0

    for i in range(height-1):
        if (frame_g[i,:] != 0).any():
            if got_start_y == False:
                start_y = i
                got_start_y = True
        else:    
            if got_start_y == True:
                height_box = i-start_y
                break
        
    for j in range(width-1):
        if (frame_g[:,j] != 0).any():
            if got_start_x == False:
                start_x = j
                got_start_x = True
        else:
            if got_start_x == True:
                width_box = j - start_x    
                break            

    # crop/trim the colored image
    frame = frame[start_y:(start_y+height_box),start_x:(start_x+width_box),:]    
    return frame


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    
    # convert both images to grayscale for feature extract and matching
    img1_g = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # create SIFT feature extractor
    sift = cv2.xfeatures2d.SIFT_create()

    # detect features from the image1 and image2
    key1, des1 = sift.detectAndCompute(img1_g, None)
    key2, des2 = sift.detectAndCompute(img2_g, None)

    key1_len = len(key1)
    key2_len = len(key2)


    ssd = np.zeros([key1_len,key2_len])
    match_arr = np.zeros([1,3])

    # compute SSD and extract matches in an array
    for i in range(key1_len):
        min_ssd_value = 99999999
        min_ssd_index = 0
        # convert to numpy array
        des1_np = np.array(des1[i])            
        for j in range(key2_len):
            # convert to numpy array
            des2_np = np.array(des2[j])
            
            # calculate SSD  
            sub = np.subtract(des1_np,des2_np)
            mul = np.multiply(sub,sub)
            ssd[i,j] = np.sum(mul)
            
            if ssd[i,j] < min_ssd_value:
                min_ssd_index = j
                min_ssd_value = ssd[i,j]


        sort_arr = np.sort(ssd[i])
        min_2nd = sort_arr[1]
        ratio = min_ssd_value/min_2nd
        if ratio < 0.8:
            match_arr = np.vstack([match_arr,np.array([i,min_ssd_index,min_ssd_value])])

    # crop the 1st row and sort according to distance in ascending order
    match_arr = match_arr[1:len(match_arr)]
    match_arr = match_arr[match_arr[:, 2].argsort()]

    # get best 500 matches and their keypoints location
    match_arr = match_arr[0:50]
    pt1 = np.zeros([len(match_arr),2],dtype=np.float32)
    pt2 = np.zeros([len(match_arr),2],dtype=np.float32)

    for i,match in enumerate(match_arr):
        pt1[i,:] = key1[int(match[0])].pt
        pt2[i,:] = key2[int(match[1])].pt


    # we need to compute homography for 1st and 2nd stitching
    h, mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,5.0)
    h1,mask1 = cv2.findHomography(pt2,pt1,cv2.RANSAC,5.0)



    # calculate modified transalation component for the 1st stitching
    trans_x = 0
    trans_y = 0
    while h[0,2] <= 0:
        h[0,2] = h[0,2] + 150
        trans_x += 150

    while h[1,2] <= 0:
        h[1,2] = h[1,2] + 150        
        trans_y += 150
    
    h[1,2] = h[1,2] + 50
    h[0,2] = h[0,2] + 50
    trans_x = trans_x + 50
    trans_y = trans_y + 50 

    # calculate modified transalation component for 2nd stitching
    trans_x1 = 0
    trans_y1 = 0
    while h1[0,2] <= 0:
        h1[0,2] = h1[0,2] + 150
        trans_x1 += 150

    while h1[1,2] <= 0:
        h1[1,2] = h1[1,2] + 150        
        trans_y1 += 150    

    # transform the 1st image and stitch the 2nd image over 1st image
    img_st = cv2.warpPerspective(img1,h,(1000,1000))
    img_st[trans_y:trans_y+img2.shape[0],trans_x:trans_x+img2.shape[1]] = img2

    # transform the 2nd image and stitch the 1nd image over 2nd image
    img_st1 = cv2.warpPerspective(img2,h1,(1000,1000))
    img_st1[trans_y1:trans_y1+img1.shape[0],trans_x1:trans_x1+img1.shape[1]] = img1

    # trim the image 
    img_st1 = trim(img_st1)

    # transform the image as it aligned with the 1st stitched image
    img_st1 = cv2.warpPerspective(img_st1,h,(10000,10000))

    # trim/crop the image to remove black boundary
    img_st = trim(img_st)
    img_st1 = trim(img_st1)

    # remove the column length mismatch due to noise
    img_st1 = img_st1[:,2:img_st1.shape[1]-2,:]

    # convert to grayscale and compute threshold - 90 worked better
    img_st_g = cv2.cvtColor(img_st,cv2.COLOR_BGR2GRAY)
    img_st1_g = cv2.cvtColor(img_st1,cv2.COLOR_BGR2GRAY)

    img_st_b = cv2.threshold(img_st_g, 90, 255, cv2.THRESH_BINARY)[1] 
    img_st1_b = cv2.threshold(img_st1_g, 90, 255, cv2.THRESH_BINARY)[1]

    # calculate image difference to create mask
    img_dif = img_st_b - img_st1_b
    # apply certain techniques to remove noises in the mask and dilate
    img_dif = cv2.medianBlur(img_dif, 5)
    kernel = np.ones((5, 5), 'uint8')
    img_dif = cv2.dilate(img_dif,kernel,iterations=1)
    
    # gaussian blur to have smoothing in the final image 
    img_dif = cv2.GaussianBlur(img_dif,(5,5),0)

    # normalise the mask
    mask = img_dif.astype(float) / 255
    
    # compute the final image
    img_fin = img_st.copy()
    for i in range(img_st.shape[0]):
        for j in range(img_st1.shape[1]):
            for k in range(3):
                img_fin[i,j,k] = int((mask[i,j])*img_st[i,j,k] + (1-mask[i,j])*img_st1[i,j,k])
    
    
    cv2.imwrite(savepath,img_fin)

    return
if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

