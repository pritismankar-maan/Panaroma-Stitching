# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

def trim(frame):
    # convert to gray scale
    frame_g = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    height,width = frame_g.shape 

    # calculate bounding box
    height_box = 0
    width_box = 0
    got_start_y = False
    got_start_x = False
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


def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    
    # create SIFT feature extractor
    sift = cv2.xfeatures2d.SIFT_create()
    img_st = imgs[0]

    for iter in range(1,len(imgs)):
        
        # covert to grayscale for feature extraction and matching
        img_sti_g = cv2.cvtColor(img_st,cv2.COLOR_BGR2GRAY)
        img_new_g = cv2.cvtColor(imgs[iter],cv2.COLOR_BGR2GRAY)

        # detect features from the image1 and image2
        key1, des1 = sift.detectAndCompute(img_sti_g, None)

        key2, des2 = sift.detectAndCompute(img_new_g, None)    

        ssd = np.zeros([len(key1),len(key2)])
        match_arr = np.zeros([1,3])

        # compute SSD and extract matches in an array
        for i in range(len(key1)):
            min_ssd_value = 99999999
            min_ssd_index = 0
            # convert to numpy array
            des1_np = np.array(des1[i])            
            for j in range(len(key2)):
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
                rows = np.where(match_arr[:,1] == min_ssd_index)[0]
                if rows.size > 0:
                    mat = match_arr[rows]
                    if min_ssd_value < mat[0,2]:
                        match_arr[rows] = [i,min_ssd_index,min_ssd_value]
                else:
                    match_arr = np.vstack([match_arr,np.array([i,min_ssd_index,min_ssd_value])])        

        
        # check if number of matches is greater than 50, go ahead and stitch
        if len(match_arr) > 55:
            # crop the 1st row
            match_arr = match_arr[1:len(match_arr)]
            match_arr = match_arr[match_arr[:, 2].argsort()]

            # select best 51 matches
            match_arr = match_arr[0:50]

            # extract corresponding keypoints of those 51 matches
            pt1 = np.zeros([len(match_arr),2],dtype=np.float32)
            pt2 = np.zeros([len(match_arr),2],dtype=np.float32)

            for i,match in enumerate(match_arr):
                pt1[i,:] = key1[int(match[0])].pt
                pt2[i,:] = key2[int(match[1])].pt


            # we need to compute homography
            h, mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,5.0)

            # calculate modified transalation component
            trans_x = 0
            trans_y = 0
            while h[0,2] < 0:
                h[0,2] = h[0,2] + 100
                trans_x += 100

            while h[1,2] < 0:
                h[1,2] = h[1,2] + 100        
                trans_y += 100
            
            # transform the 1st image
            img_st = cv2.warpPerspective(img_st,h,(10000,10000))

            # stitch the next image
            img_st[trans_y:trans_y+imgs[iter].shape[0],trans_x:trans_x+imgs[iter].shape[1]] = imgs[iter]

            # remove black boundaries
            img_st = trim(img_st)
    
    cv2.imwrite(savepath,img_st)

    # initilize overlapping array
    overlap_arr = np.eye(4)
    
    # compare each image with other images
    for i_iter in range(len(imgs)):
        for j_iter in range(len(imgs)):
            if i_iter != j_iter:
                # covert to grayscale for feature extraction and matching
                img_sti_g = cv2.cvtColor(imgs[i_iter],cv2.COLOR_BGR2GRAY)
                img_new_g = cv2.cvtColor(imgs[j_iter],cv2.COLOR_BGR2GRAY)

                # detect features from the image1 and image2
                key1, des1 = sift.detectAndCompute(img_sti_g, None)

                key2, des2 = sift.detectAndCompute(img_new_g, None)    

                ssd = np.zeros([len(key1),len(key2)])
                match_arr = np.zeros([1,3])

                # compute SSD and extract matches in an array
                for i in range(len(key1)):
                    min_ssd_value = 99999999
                    min_ssd_index = 0
                    # convert to numpy array
                    des1_np = np.array(des1[i])            
                    for j in range(len(key2)):
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
                        rows = np.where(match_arr[:,1] == min_ssd_index)[0]
                        if rows.size > 0:
                            mat = match_arr[rows]
                            if min_ssd_value < mat[0,2]:
                                match_arr[rows] = [i,min_ssd_index,min_ssd_value]
                        else:
                            match_arr = np.vstack([match_arr,np.array([i,min_ssd_index,min_ssd_value])])

                # crop the 1st row
                match_arr = match_arr[1:len(match_arr)]

                # get overlap percentage and update the matrix
                len_good_matches = len(match_arr)
                overlap_perc = len_good_matches/len(key1)

                if overlap_perc > 0.2:
                    overlap_arr[i_iter,j_iter] = 1


    return overlap_arr
if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
