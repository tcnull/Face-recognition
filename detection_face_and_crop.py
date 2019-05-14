import cv2

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face
def main():
    img_dir='./images/test_img'
    img_path_set=[]     #path of every images
    for file in os.listdir(img_dir):
        single_img=os.path.join(img_dir,file)
        print('loading: {}'.format(file))
        img_path_set.append(single_img)

    images = load_and_align_data(img_path_set, 160, 44)
    emb_dir='./images/emb_img'
    
    if(os.path.exists(emb_dir)==False):
        os.mkdir(emb_dir)

    count=0
    for file in os.listdir(img_dir):
        print("save {} ".format(file))
        misc.imsave(os.path.join(emb_dir,file),images[count])
        count=count+1
    print("get {} faces".format(count))
    
def load_and_align_data(image_paths, image_size, margin):

    minisize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # threshold of bounding_boxe in three nets
    factor = 0.709 # scale factor of face pyramid
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        # apply video memory dynamically
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # increase slowly until to max capacity when use GPU 
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4 # use 40% capacity of GPU
        sess = tf.Session(config=config)
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        print(image)
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minisize, pnet, rnet, onet, threshold, factor)
        print(bounding_boxes)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue

        det = np.squeeze(bounding_boxes[0,0:4])
        
        bb = np.zeros(4, dtype=np.int32)

        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]

        # resize images deal with alignd
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


if __name__=='__main__':
    main()