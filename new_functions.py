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
import math
from sklearn.model_selection import train_test_split
import torch
import torchvision
########################################################################


#########################################################################   
#                               Zarins's Code                           #                                                               #
#########################################################################   

def resize2(path):
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
            imResize = im.resize((448,448), Image.ANTIALIAS)
            print(imResize)
            imResize.show()
            imResize.save('C:/Users/zarin/Desktop/abc' + str(c)+' resized.jpg', 'JPEG', quality=90)
            if c == 3:
                break

        #imResize.save(f + ' resized.jpg', 'JPEG', quality=90) <------------f is the path can change it to have a good name
    return c


#def gen_labels(json_dict, g_size, Positive = True):
def gen_labels2(Y, Joints, grid_size=3):
    """
    Inputs:
         Y := numpy array of shape (N, ) indicating if hand is present
         Joints := numpy array of shape (N, 21, 2) of joint coordinates

    Outputs:
        Labels := numpy array of shape (N, grid_size, grid_size, len(y))
    """
    # grid_size = 3
    # v_length = 5
    # labels = torch.randn(10,grid_size, grid_size, v_length)
    N = len(Y)
    labels = np.zeros((N, grid_size, grid_size, 5))
   
    for c, joints in enumerate(Joints):
        min_x, min_y = joints.min(axis=0) / np.array([1920, 1080])
        max_x, max_y = joints.max(axis=0) / np.array([1920, 1080])
        # print('min_x', min_x)
        # print('max_x', max_x)
        # print('min_y', min_y)
        # print('max_y', max_y)

        height = max_y - min_y
        width = max_x - min_x
        center_x = min_x + width / 2.
        center_y = min_y + height / 2.

        # print('height', height)
        # print('width', width)
        # print('center_x', center_x)
        # print('center_y', center_y)

        grid_idx_x = math.floor(center_x * grid_size) 
        grid_idx_y = math.floor(center_y * grid_size)
        center_x_new = ((center_x*100) - (grid_idx_x* (100/3.)))/ (100/3.)
        center_y_new = ((center_y*100) - (grid_idx_y* (100/3.)))/ (100/3.)
        p_hand = 0
        p_nohand = 0
        # (pc, x, y, w, h, p(hand/object), p(nohand/object)
        if Y[c] == 1:
            #hand is present
            p_hand = 1
        else:
            p_nohand = 1
            
        #iou = iou ([center_x_new, center_y_new, width, height], 
        labels[c][grid_idx_y][grid_idx_x] = np.array([Y[c], center_x_new, center_y_new, width, height])
        #labels[c][grid_idx_y][grid_idx_x] = np.array([Y[c], center_x_new, center_y_new, width, height, p_hand, p_nohand])
        #labels[c][grid_idx_y][grid_idx_x] = np.array([Y[c], 0, 0, 0 ,0])

    return labels


#########################################################################   
#                               John's Code                             #                                                               #
#########################################################################   
import imageio
from PIL import Image
import json
import numpy as np
from matplotlib import pyplot as plt

def read_img_data2(N, img_height=100, img_width=100):
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
            
         return X, Y, Joint_Coords, Hand_Info
    """  

    N_pos = N*.5  # num positive images to read
    N_neg = N - N_pos  # num negative images to read
    X = np.zeros((N, img_height, img_width, 3))
    Y = np.zeros((N,))
    Joint_Coords = np.zeros((N, 21, 2))
    Hand_Info = np.zeros((N, ))

    # Read annotations.json
    J = read_joints2()

    # Read Positive Images
    img_dir = POS_IMAGES_PATH
    c = 0
    for item in os.listdir(img_dir):
        if c >= N_pos:
            break

        # Check if left or right hand
        f, ext = os.path.splitext(item)
        # print(f)
        if (f+'_L' in J):
            # assert (f+'_R' not in J), f+'_R'+' also found in J'
            # print('Left Hand')
            Hand_Info[c] = 0.
            Joint_Coords[c] = J[f+'_L']
        elif (f+'_R' in J):
            assert (f+'_L' not in J), f+'_L'+' also found in J'
            # print('Right Hand')
            Hand_Info[c] = 1.
            Joint_Coords[c] = J[f+'_R']
        else:
            print('Not Left nor Right!')

        # Read the image and resize. Store as numpy array.
        img_file = os.path.join(img_dir, item)
        if os.path.isfile(img_file):
            #print("pos")
            im = Image.open(img_file)
            im_resized = im.resize((img_height, img_width), Image.ANTIALIAS)
            X[c] = np.array(im_resized)
            Y[c] = 1.
            c += 1

    # Read Negative Images
    neg_img_dir = NEG_IMAGES_PATH
    for item in os.listdir(neg_img_dir):
        if c >= N:
            break
        # Read the image and resize. Store as numpy array.
        neg_img_file = os.path.join(neg_img_dir, item)
        if os.path.isfile(neg_img_file):
            #print(neg_img_file)
            im = Image.open(neg_img_file)
            im_resized = im.resize((img_height, img_width), Image.ANTIALIAS)
            X[c] = np.array(im_resized)
            Y[c] = 0.
            c += 1



    return X, Y, Joint_Coords, Hand_Info


def read_joints2():
    """
    Assumes annotations.json in the current directory. 

    Outputs:
    joint_dict := dictionary of joint locations. 
                      Format { "im name" : numpy array (21, 2)}

    """
    with open('annotation.json') as f:
        joint_data = json.load(f)

    return joint_data



def split2(X, Y, N):
    #N = Number of images in total
    values = np.random.permutation(N)
    new_x = X[values]
    new_y = Y[values]
    
    x, X_test, y, Y_test =  train_test_split(new_x, new_y, test_size = .20, random_state = 42)
    X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size = .20, random_state = 42)
    
    return X_train, X_test,X_val, Y_val, Y_train, Y_test


############################# Plotting Functions #########################
def plot_bounding_box2(img, cx, cy, w, h, hand=None):
    """
    Inputs:
        img := numpy array of shape (img_height, img_width, n_channels)
        cx, cy, w, h := coordinates of hand and width and height (0,1) scale
        hand := If not none, then either 0 or 1 indicating which hand. 0 for 'left' 1 for 'right'
    """
    
    
    img_h, img_w = img.shape[0], img.shape[1]
    box_top_left_x, box_top_left_y = (cx-w/2)*img_w, (cy-h/2)*img_h
    box_h, box_w = h*img_h, w*img_w
    fig, ax = plt.subplots()
    plt.imshow(img.astype(np.uint8))
    plt.plot(cx*img_w, cy*img_h, color='g', marker='o', markersize='2')
    r = plt.Rectangle((box_top_left_x, box_top_left_y), box_w, box_h, edgecolor='r', facecolor='none')
    ax.add_artist(r)

    # Plot hand label
    if hand == 0:
        fig.text(cx, cy, 'Left', fontsize=8, bbox=dict(facecolor='gray', alpha=0.5))
    elif hand == 1:
        fig.text(cx, cy, 'right', fontsize=8, bbox=dict(facecolor='gray', alpha=0.5))

    plt.show()
    
def read_img_dataC(N, img_height=100, img_width=100):
    N_pos = N/2.  # num positive images to read
    N_neg = N - N_pos  # num negative images to read
    X = np.zeros((N, img_height, img_width, 3))
    Y = np.zeros((N,))
    Joint_Coords = np.zeros((N, 21, 2))
    Hand_Info = np.zeros((N, ))

    # Read annotations.json
    J = read_joints2()

    # Read Positive Images
    img_dir = POS_IMAGES_PATH
    c = 0
    for item in os.listdir(img_dir):
        if c >= N_pos:
            break

        # Check if left or right hand
        f, ext = os.path.splitext(item)
        # print(f)
        if (f+'_L' in J):
            # assert (f+'_R' not in J), f+'_R'+' also found in J'
            #print('Left Hand')
            Hand_Info[c] = 0.
            Joint_Coords[c] = J[f+'_L']
        elif (f+'_R' in J):
            assert (f+'_L' not in J), f+'_L'+' also found in J'
            #print('Right Hand')
            Hand_Info[c] = 1.
            Joint_Coords[c] = J[f+'_R']
        else:
            print('Not Left nor Right!')

        # Read the image and resize. Store as numpy array.
        min_x, min_y = Joint_Coords[c].min(axis=0) 
        max_x, max_y = Joint_Coords[c].max(axis=0) 
        # print('min_x', min_x)
        # print('max_x', max_x)
        # print('min_y', min_y)
        # print('max_y', max_y)

        height = max_y - min_y
        width = max_x - min_x
        center_x = min_x + width / 2.
        center_y = min_y + height / 2.
        img_file = os.path.join(img_dir, item)
        #print(min_x*1920, min_y*1080)
        if os.path.isfile(img_file):
            #print("pos")
            im = Image.open(img_file)
            
            im_cropped = torchvision.transforms.functional.crop(im, min_y-50, min_x-50, 300, 300)
#        imResize = im.resize((100,100), Image.ANTIALIAS)
            #print(im_cropped)
            #im_cropped.show()
            im_resized = im_cropped.resize((img_height, img_width), Image.ANTIALIAS)
            #im_resized.show()
            X[c] = np.array(im_resized)
            Y[c] = 1.
            c += 1

    # Read Negative Images
    neg_img_dir = NEG_IMAGES_PATH
    for item in os.listdir(neg_img_dir):
        if c >= N:
            break
        # Read the image and resize. Store as numpy array.
        neg_img_file = os.path.join(neg_img_dir, item)
        if os.path.isfile(neg_img_file):
            #print(neg_img_file)
            im = Image.open(neg_img_file)
            im_resized = im.resize((img_height, img_width), Image.ANTIALIAS)
            X[c] = np.array(im_resized)
            Y[c] = 0.
            c += 1



    return X, Y, Joint_Coords, Hand_Info

def iou(boxA, boxB):
    #boxA = [x- bottom left, y bottom left, x top right, y top right]               
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])*1920
    yA = max(boxA[1], boxB[1])*1080
    xB = min(boxA[2], boxB[2])*1920
    yB = min(boxA[3], boxB[3])*1080

    # compute the area of intersection rectangle
    interArea = max(0,(xB - xA + 1)) * max(0,(yB - yA + 1))
    #interArea = max(0, xB – xA + 1) * max(0, yB – yA + 1)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou_val = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou_val

############################### Other Functions ##########################
def to_channels_first2(X):
    """
    Inputs:
        X := numpy array of images in channels last order (N, h, w, channels)
    Outputs:
        X_rolled := Channels first array of images (channels, N, h, w)
    """
    X_rolled = np.rollaxis(X, axis=3)
    X_rolled = np.rollaxis(X_rolled, axis=1)
    return X_rolled

def unpickle2(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def gen_negative_images2():
    cifar10_dict = unpickle('cifar-10-batches-py/data_batch_1')
    cifar_imgs = cifar10_dict[b'data']
    cifar_imgs = cifar_imgs.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    for i,img in enumerate(cifar_imgs):
        im = Image.fromarray(img)
        im.save('negative_images/'+str(i)+'.jpg')