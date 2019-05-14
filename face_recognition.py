from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cv2 import cv2
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import facenet
import align.detect_face
import detection_face_and_crop

minsize = 20
threshold = [ 0.6, 0.7, 0.7 ]
factor = 0.709 

def load_and_align_data(img, image_size, margin):
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    if len(bounding_boxes) < 1:
        return 0,0,0
    det=bounding_boxes
    det[:,0]=np.maximum(det[:,0], 0)
    det[:,1]=np.maximum(det[:,1], 0)
    det[:,2]=np.minimum(det[:,2], img_size[1])
    det[:,3]=np.minimum(det[:,3], img_size[0])

    det=det.astype(int)
    crop=[]
    for i in range(len(bounding_boxes)):
        temp_crop=img[det[i,1]:det[i,3],det[i,0]:det[i,2],:]
        aligned=misc.imresize(temp_crop, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        crop.append(prewhitened)

    crop_image=np.stack(crop)  
    return 1,det,crop_image


def main():

    detection_face_and_crop.main()

    with tf.Graph().as_default():
        with tf.Session() as sess:     
            model='./20170512-110547/'
            # model='./20180408-102900/'
            facenet.load_model(model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image=[]
            nrof_images=0

            emb_dir='./images/emb_img'
            all_obj=[]
            for i in os.listdir(emb_dir):
                all_obj.append(i)
                img = misc.imread(os.path.join(emb_dir,i), mode='RGB')
                prewhitened = facenet.prewhiten(img)
                image.append(prewhitened)
                nrof_images=nrof_images+1

            images=np.stack(image)
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            # get metrixs of images in emb_img
            compare_emb = sess.run(embeddings, feed_dict=feed_dict) 
            compare_num=len(compare_emb)


            # video="http://admin:admin@192.168.137.33:8081/"
            # capture =cv2.VideoCapture(video)
            dirVideo = "video1.mp4"
            capture =cv2.VideoCapture(dirVideo)
            # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            capture.set(cv2.CAP_PROP_FPS, 60)
            
            # size =(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            # writeVideo = cv2.VideoWriter("aaa.avi", fourcc, 5, size)
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writeVideo = cv2.VideoWriter('output.avi',fourcc, 20, size, 1)

            cv2.namedWindow("camera",1)
            picNumber = 0
            count = 0
            frame_interval = 3
            while True:
                isSuccess, frame = capture.read() 
                if(count % frame_interval == 0):
                    rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tag, bounding_box, crop_image, =load_and_align_data(rgb_frame,160,44)
                    if(tag):
                        feed_dict = { images_placeholder: crop_image, phase_train_placeholder:False }
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        print(emb)
                        temp_num=len(emb)
                        fin_obj=[]
                        # calculate distance between camera face and in emd_img face
                        for i in range(temp_num):
                            dist_list=[]
                            for j in range(compare_num):
                                dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], compare_emb[j,:]))))
                                
                                dist_list.append(dist)
                            min_value=min(dist_list)
                            if(min_value>0.65):
                                fin_obj.append('UNKNOW')
                            else:
                                fin_obj.append(all_obj[dist_list.index(min_value)][0:6])    #mini distance is face which recongnition
                        # draw rectangle
                        for rec_position in range(temp_num):                        
                            cv2.rectangle(frame,
                                            (bounding_box[rec_position,0],
                                            bounding_box[rec_position,1]),
                                            (bounding_box[rec_position,2],
                                            bounding_box[rec_position,3]),
                                            (0, 255, 0), 2, 8, 0)
                            cv2.putText(frame,
                                        fin_obj[rec_position], 
                                        (bounding_box[rec_position,0],bounding_box[rec_position,1]),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                                        0.8, 
                                        (0, 0 ,255), 
                                        thickness = 2, 
                                        lineType = 2)
                    writeVideo.write(frame)
                    cv2.imshow('camera',frame)
                count += 1
                key = cv2.waitKey(3)
                if key == 27:
                    print("ESC break")
                    break
                if key == ord(' '):
                    picNumber += 1
                    # filename = "{}_{}.jpg".format(dirVideo, picNumber)
                    filename = "%s_%s.jpg" % (dirVideo, picNumber)
                    cv2.imwrite(filename,frame)
            capture.release()
            cv2.destroyWindow("camera")

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

if __name__=='__main__':
    main()