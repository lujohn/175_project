############################## Files Structure #########################
# "current directory" is the directory of your jupyter-notebook
# positve_images := folder for positive images
# negative_images := folder for negative imgages
# annotations.json := in current directory

POS_IMAGES_PATH = 'positive_images_resized'
NEG_IMAGES_PATH = 'negative_images'
SAMPLE_IMAGES_PATH = 'sample_images'

import os
import sys
import math
from sklearn.model_selection import train_test_split



IMG_WIDTH = 1920
IMG_HEIGHT = 1080

########################################################################


#########################################################################   
#                               Zarins's Code                           #                                                               #
#########################################################################   

#def gen_labels(json_dict, g_size, Positive = True):
def gen_labels(Y, Joints, grid_size=3):
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
    #json_dict = {"img1":  [(1,2), (3,4), (5,6),(7,8), (9,6), (4,67), (23,4), (30,45), ()]}
    #Assuming that joints are in order
    # for image in list_:            #better if image is a number
    #     left = json[image][17]
    #     right = json[image][2]
    #     top = json[image][12]
    #     down = json[image][0]
    #     bx = (right-left)/2
    #     by = (down -top)/2
    #     bw= abs(left - right)
    #     bh = abs(top-down)
    #     labels[image]

    grid_width = IMG_WIDTH / grid_size
    grid_height = IMG_HEIGHT / grid_size
    for c, joints in enumerate(Joints):
        # Get min and max locations from raw image
        min_x, min_y = joints.min(axis=0)
        max_x, max_y = joints.max(axis=0)

        # Calculate width, height, and center
        height = max_y - min_y
        width = max_x - min_x
        center_x = min_x + width / 2.
        center_y = min_y + height / 2.

        # Assign a grid responsible
        grid_idx_x = math.floor(center_x / grid_width)
        grid_idx_y = math.floor(center_y / grid_height)

        # Scale width and height
        width_scaled = width / IMG_WIDTH
        height_scaled = height / IMG_HEIGHT

        # Scale coordinates to be in coordinate system of grid responsible
        x_start = grid_idx_x * (IMG_WIDTH / grid_size)
        y_start = grid_idx_y * (IMG_HEIGHT / grid_size)
        center_x_scaled = (center_x - x_start) / grid_width
        center_y_scaled = (center_y - y_start) / grid_height

        labels[c][grid_idx_y][grid_idx_x] = np.array([1, center_x_scaled, center_y_scaled, width_scaled, height_scaled])



        # min_x, min_y = joints.min(axis=0) / np.array([IMG_WIDTH, IMG_HEIGHT])
        # max_x, max_y = joints.max(axis=0) / np.array([IMG_WIDTH, IMG_HEIGHT])
        # # print('min_x', min_x)
        # # print('max_x', max_x)
        # # print('min_y', min_y)
        # # print('max_y', max_y)

        # height = max_y - min_y
        # width = max_x - min_x
        # center_x = min_x + width / 2.
        # center_y = min_y + height / 2.

        # # print('height', height)
        # # print('width', width)
        # # print('center_x', center_x)
        # # print('center_y', center_y)

        # # Grid Responsible
        # grid_idx_x = math.floor(center_x * grid_size) 
        # grid_idx_y = math.floor(center_y * grid_size)

        # labels[c][grid_idx_y][grid_idx_x] = np.array([Y[c], center_x, center_y, width, height])

    return labels


#########################################################################   
#                               John's Code                             #                                                               #
#########################################################################   
import imageio
from PIL import Image
import json
import numpy as np
from matplotlib import pyplot as plt

def read_img_data(N, img_height=100, img_width=100, print_every=500):
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

    N_pos = N // 2  # num positive images to read
    N_neg = N - N_pos  # num negative images to read
    X = np.zeros((N, img_height, img_width, 3))
    Y = np.zeros((N,))
    Joint_Coords = np.zeros((N, 21, 2))
    Hand_Info = np.zeros((N, ))

    # Read annotations.json
    J = read_joints()

    # Read Positive Images
    img_dir = POS_IMAGES_PATH
    c = 0
    for i, item in enumerate(os.listdir(img_dir)):
        if c >= N_pos:
            break

        if i % print_every == 0:
            print("Loading Positive Image: ", i)

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
            im = Image.open(img_file)
            im_resized = im.resize((img_height, img_width), Image.ANTIALIAS)
            X[c] = np.array(im_resized)
            Y[c] = 1.
            c += 1

    # Read Negative Images
    neg_img_dir = NEG_IMAGES_PATH
    for i, item in enumerate(os.listdir(neg_img_dir)):
        if c >= N:
            break

        if i % print_every == 0:
            print("Loading Negative Image: ", i)

        # Read the image and resize. Store as numpy array.
        neg_img_file = os.path.join(neg_img_dir, item)
        if os.path.isfile(neg_img_file):
            im = Image.open(neg_img_file)
            im_resized = im.resize((img_height, img_width), Image.ANTIALIAS)
            X[c] = np.array(im_resized)
            Y[c] = 0.
            c += 1



    return X, Y, Joint_Coords, Hand_Info


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



def split(X, Y, N):
    #N = Number of images in total
    values = np.random.permutation(N)
    new_x = X[values]
    new_y = Y[values]
    
    x, X_test, y, Y_test =  train_test_split(new_x, new_y, test_size = .20, random_state = 42)
    X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size = .20, random_state = 42)
    
    return X_train, X_test,X_val, Y_val, Y_train, Y_test


############################# Plotting Functions #########################
def plot_bounding_box_from_grid(img, grid):
    s = grid.shape[0] # grid_size
    img_h, img_w = img.shape[0], img.shape[1]
    for i in range(s):
        for j in range(s):
            if grid[j][i][0] == 1:
                _, cx, cy, w, h, *_ = grid[j][i]
                cx = (i+cx)*(img_w/s)
                cy = (j+cy)*(img_h/s)
                w = w*img_w
                h = h*img_h
                plot_bounding_box(img, cx, cy, w, h)

def plot_bounding_box(img, cx, cy, w, h, hand=None):
    """
    Inputs:
        img := numpy array of shape (img_height, img_width, n_channels)
        cx, cy, w, h := coordinates of hand and width and height (0,1) scale
        hand := If not none, then either 0 or 1 indicating which hand. 0 for 'left' 1 for 'right'
    """
    box_top_left_x, box_top_left_y = (cx-w/2), (cy-h/2)
    box_h, box_w = h, w
    fig, ax = plt.subplots()
    plt.imshow(img.astype(np.uint8))
    plt.plot(cx, cy, color='g', marker='o', markersize='2')
    r = plt.Rectangle((box_top_left_x, box_top_left_y), box_w, box_h, edgecolor='r', facecolor='none')
    ax.add_artist(r)

    # Plot hand label
    if hand == 0:
        fig.text(cx, cy, 'Left', fontsize=8, bbox=dict(facecolor='gray', alpha=0.5))
    elif hand == 1:
        fig.text(cx, cy, 'right', fontsize=8, bbox=dict(facecolor='gray', alpha=0.5))

    plt.show()
############################### Other Functions ##########################
def to_channels_first(X):
    """
    Inputs:
        X := numpy array of images in channels last order (N, h, w, channels)
    Outputs:
        X_rolled := Channels first array of images (channels, N, h, w)
    """
    X_rolled = np.rollaxis(X, axis=3)
    X_rolled = np.rollaxis(X_rolled, axis=1)
    return X_rolled

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def gen_negative_images():
    cifar10_dict = unpickle('cifar-10-batches-py/data_batch_1')
    cifar_imgs = cifar10_dict[b'data']
    cifar_imgs = cifar_imgs.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    for i,img in enumerate(cifar_imgs):
        im = Image.fromarray(img)
        im.save('negative_images/'+str(i)+'.jpg')