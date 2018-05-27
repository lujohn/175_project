############################## Files Structure #########################
# "current directory" is the directory of your jupyter-notebook
# positve_images := folder for positive images
# negative_images := folder for negative imgages
# annotations.json := in current directory

POS_IMAGES_PATH = 'positive_images'
NEG_IMAGES_PATH = 'negative_images'
SAMPLE_IMAGES_PATH = 'sample_images'

import os
import sys
########################################################################


#########################################################################   
#                               Zarins's Code                           #                                                               #
#########################################################################   

def resize(path):
    #path = "C:/Users/zarin/Desktop/Dataset/Color/"
    dirs = os.listdir( path )
    c =0
    for item in dirs:
        if os.path.isfile(path+item):
            c+=1
            im = Image.open(path+item)
            print(im.size)
            f, e = os.path.splitext(path+item)
            print(f)
            print(e)
            #break
            imResize = im.resize((100,100), Image.ANTIALIAS)
            print(imResize)
            imResize.show()
            imResize.save('C:/Users/zarin/Desktop/abc' + str(c)+' resized.jpg', 'JPEG', quality=90)
            if c == 3:
                break

        #imResize.save(f + ' resized.jpg', 'JPEG', quality=90) <------------f is the path can change it to have a good name
    return c


#def gen_labels(json_dict, g_size, Positive = True):
def gen_labels():
    grid_size = 3
    v_length = 5
    labels = torch.randn(10,grid_size, grid_size, v_length)
    #json_dict = {"img1":  [(1,2), (3,4), (5,6),(7,8), (9,6), (4,67), (23,4), (30,45), ()]}
    #Assuming that joints are in order
    for image in list_:            #better if image is a number
        left = json[image][17]
        right = json[image][2]
        top = json[image][12]
        down = json[image][0]
        bx = (right-left)/2
        by = (down -top)/2
        bw= abs(left - right)
        bh = abs(top-down)
        labels[image]
        
    
    return labels


#########################################################################   
#                               John's Code                             #                                                               #
#########################################################################   
import imageio
from PIL import Image
import json
import numpy as np
from matplotlib import pyplot as plt

def read_img_data(N, img_height=100, img_width=100):
    """
        Inputs:
         N := number of images to read
         img_height, img_width := height and width of images to resize to

        Outputs:
         X := Numpy Array of shape (N, img_height, img, width, n_channels).
         Y := Numpy vector of shape (N, ) indicating whether a hand is present or not.
         J := Numpy Array of joint information of shape: (N, 21, 2).
         Hand_Info := Numpy vector indicating which hand (Left or Right) of shape (N, )
                          Hand_Info[i] = 0 if left and 1 if right
                          
        Pseudocode:
         X = empty numpy array of shape (N, img_height, img_width, n_channels)
         Joint_Coords = empty numpy array of shape (N, num_joints, 2)
         Hand_Info = empty numpy array of shape (N, )
         
         joint_dict = read_joints_json()
         
         i = 0
         for each img in "positve" images folder:
            img_name = get image name from file name
            img_num, hand = split img_name in to number and "which hand"
            
            X[i] = img
            Y[i] = 1
            Joint_Coords[i] = joint_dict[img_name]
            Hand_Info[i] = hand
            i += 1
         
         for each img in "negative" images folder:
            img_name = get image name from file name
            img_num, hand = split img_name in to number and "which hand"
            
            X[i] = img
            Y[i] = 0
            Joint_Coords[i] = bunch of don't cares
            Hand_Info[i] = -10 # Don't Care for negative images
            
         return X, Joint_Coords, Hand_Info
    """  
    X = np.zeros((N, img_height, img_width, 3))
    Y = np.zeros((N,))
    Hand_Info = np.zeros((N, ))

    # Read annotations.json
    J = read_joints()

    img_dir = os.path.join(POS_IMAGES_PATH)
    c = 0
    for item in os.listdir(img_dir):
        if c >= N:
            break

        # Check if left or right hand
        f, ext = os.path.splitext(item)
        if (f+'_L' in J):
            print('Left Hand')
            Hand_Info[c] = 0.
        elif (f+'_R' in J):
            print('Right Hand')
            Hand_Info[c] = 1.
        else:
            print('Not Left nor Right!')

        # Read the image and resize. Store as numpy array.
        img_file = os.path.join(img_dir, item)
        if os.path.isfile(img_file):
            im = Image.open(img_file)
            im_resized = im.resize((img_height, img_width), Image.ANTIALIAS)
            X[c] = np.array(im_resized)
            Y[c] = 1.
            c += 1

    #### TODO: Implement above for Negatives Folder ####

    return X, Y, Hand_Info


def read_joints():
    """
    Assumes annotations.json in the current directory. 

    Outputs:
    joint_dict := dictionary of joint locations. 
                      Format { "im name" : numpy array (21, 2)}

    """
    with open('annotation.json') as f:
        joint_data = json.load(f)

    return joint_data

 

def split_data(X):
    """
    blah
    """