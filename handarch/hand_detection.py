
"""

Object Detection Classes

Author: Gabriel Peixoto de Carvalho
Requirements:
python 3.6.5
opencv 3.4.1
Tensorflow 1.8

DESCRIPTION:
This class has the following methods implemented:
    1. Cascade (Viola-Jones) detection/tracking
    2. Color Detection (Trained EM segmentation + largest contour)
    3. Movement Detection (MOG2 background subtractor + largest contour)
    4. CNN (Tensorflow object Detection API trained with egohands dataset, single and multithread)

These classes receive an image and return the hand detected square coordinates in the shape: x,y,w,h

@# TODO: definir metrica de confiabilidade da deteccao

 LICENSE:
 not decided

"""



import cv2 as cv
import re
import numpy as np
from time import time
import os
import pickle
import image_utils as imutils
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import label_map_util
from collections import defaultdict
import multiprocessing
from multiprocessing import Queue, Pool

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
#sess = tf.Session(config=config)

class cascadeDetection:
    """
        Hand detection using Viola-Jones Framework

        Attributes:
            cascadeXML (file): xml file which contains the trained cascade
            cascade (object): cascade classifier object, instanced using the cascadeXML file

    """
    def __init__(self, cascadeXML):
        """
            The __init__ method creates the cascade detector object

            Args:
                cascadeXML (file): xml file which contains the trained cascade

            Attributes:
                cascadeXML (file): xml file which contains the trained cascade
                cascade (object): cascade classifier object, instanced using the cascadeXML file
        """
        self.cascadeXML = cascadeXML
        self.cascade = cv.CascadeClassifier(self.cascadeXML)

    def debug(self,image,x,y,w,h):
        """
            helper function to extract the desired region of interest (ROI)
            from a source image and show it in a separated window.

            Args:
                image (np array [w,h,3]): source image
                x (int): x coordinate of the top-left point of the ROI
                y (int): y cordinate of the top-left point of the ROI
                w (int): width of the ROI
                h (int): height of the ROI

            Attributes:
                roi_color (numpy array [w,h,3]): extracted ROI

        """
        roi_color = image[y:y+h, x:x+w]
        cv.imshow("hand",roi_color)

    def detect(self, image):
        """
            detection method, uses Viola-Jones Object detection Framework to detect a hand in a specific pose

            Args:
                image (numpy array[w,h,3]): source image

            Attributes:
                gray (numpy array[w,h,1]): grayscale converted source image
                hands (): detected hands
                box (int list): detected hand ROI position information

            Return:
                box (int list): detected hand ROI position information
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        hands = self.cascade.detectMultiScale(gray, 1.3, 5)
        #count = 0
        box = None
        for (x,y,w,h) in hands:
            #count+=1
            #print("hand: ",count)
            #cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            #self.debug(image,x,y,w,h)
            box = x,y,w,h
        return box

class colorDetection:
    def __init__(self, dir):
        self.covs_path = dir+"cov_em.dat"
        self.means_path = dir+"mean_em.dat"
        #print(self.covs_path)
        self.means = pickle.load( open( self.means_path, "rb" ) )
        self.covs = pickle.load( open( self.covs_path, "rb" ) )
        #print(self.means)

    def FindHand(self,image):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
        image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
        #area,image,x,y,w,h = imutils.FindBiggestContour(image)
        info = imutils.FindBiggestContour(image)

        if info is None:
            return None
        else:
            area,image,x,y,w,h = info

        #print(area)
        if area > 0 and (w > 0 or h > 0):
            box = x,y,w,h
        else:
            box = None
        return box

    #substitui EMinferSegmentation
    def detect(self,img, no_of_clusters=2):
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        output = img.copy()
        colors = np.array([[255, 255, 255], [0, 0, 0]])
        x, y, z = img.shape
        distance = [0] * no_of_clusters
        t0 = time()
        for i in range(x):
            for j in range(y):
                for k in range(no_of_clusters):
                    diff = img[i, j] - self.means[k]
                    distance[k] = abs(np.dot(np.dot(diff, self.covs[k]), diff.T))
                output[i][j] = colors[distance.index(max(distance))]
        #print("Segmentation done in %0.3fs" % (time() - t0))
        box = self.FindHand(output)
        #cv.imshow('Largest Contour', im)
        return box

class movementDetection:
    def __init__(self):
            self.history = 30
            self.varThreshold = 16
            self.bShadowDetection = False
            self.fgbg = cv.createBackgroundSubtractorMOG2(self.history, self.varThreshold, self.bShadowDetection)

    def FindHand(self,mask,img):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
        #area,image,x,y,w,h = imutils.FindBiggestContour(image)
        info = imutils.FindBiggestContour(image)
        if info is None:
            return None
        else:
            area,image,x,y,w,h = info
        roi = imutils.getROI(img,x,y,w,h)
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        #output = cv.add(roi,image)
        #output = roi & image;
        print("area = ",area)
        if area > 25000 and area < 30000 and (w > 0 or h > 0):
            box = x,y,w,h
        else:
            box = None
        return box

    def detect(self,img):
        fgmask = self.fgbg.apply(img, None, 0.001)
        box = self.FindHand(fgmask.copy(),img)
        #cv.imshow('hand',im)
        return box

class cnnDetectionSingleThread:

    def __init__(self, model_dir, label_path):
        #self.detection_graph = tf.Graph()
        sys.path.append("..")

        # score threshold for showing bounding boxes.
        self.score_thresh = 0.27
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.chkpt_path = model_dir + '/frozen_inference_graph.pb'
        # List of the strings that is used to add correct label for each box.
        self.label_path = label_path

        self.num_classes = 1
        # load label map
        self.label_map = label_map_util.load_labelmap(self.label_path)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.detection_graph, self.sess = self.load_inference_graph()


    # Load a frozen infrerence graph into memory
    def load_inference_graph(self):
        # load frozen tensorflow model into memory
        print("> ====== loading HAND frozen graph into memory")
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.io.gfile.GFile(self.chkpt_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph, config=config)
        print(">  ====== Hand Inference graph loaded.")
        return detection_graph, sess


    # Actual detection .. generate scores and bounding boxes given an image
    def detect_objects(self, image_np, detection_graph, sess):
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name(
            'detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name(
            'num_detections:0')

        image_np_expanded = np.expand_dims(image_np, axis=0)

        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores,
                detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        #print(boxes)
        return np.squeeze(boxes), np.squeeze(scores)

    def detect(self, image_np):
        num_hands_detect = 1
        im_height, im_width = image_np.shape[:2]
        image_np = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)
        # actual detection
        boxes, scores = self.detect_objects(image_np, self.detection_graph, self.sess)

        # draw bounding boxes
        #imutils.draw_box_on_image(num_hands_detect, self.score_thresh, scores, boxes, im_width, im_height, image_np)
        box = imutils.getWindow(num_hands_detect, self.score_thresh, scores, boxes, im_width, im_height, image_np)
        #print("box = ", box)
        #image_np = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)
        return box

class cnnDetectionMultiThread:
    def __init__(self, model_dir, label_path):
        #self.detection_graph = tf.Graph()
        sys.path.append("..")

        # score threshold for showing bounding boxes.
        self.score_thresh = 0.27
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.chkpt_path = model_dir + '/frozen_inference_graph.pb'
        # List of the strings that is used to add correct label for each box.
        self.label_path = label_path
        self.num_hands_detect = 2

        self.num_classes = 1
        # load label map
        self.label_map = label_map_util.load_labelmap(self.label_path)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        #self.detection_graph, self.sess = self.load_inference_graph()
        self.nworkers = 2#multiprocessing.cpu_count()
        print(self.nworkers)
        self.input_q = Queue(maxsize=self.nworkers)
        self.output_q = Queue(maxsize=self.nworkers)
        #self.initPool()
        self.pool = Pool(self.nworkers, self.worker,(self.input_q, self.output_q))

    # Load a frozen infrerence graph into memory
    def load_inference_graph(self):
        # load frozen tensorflow model into memory
        print("> ====== loading HAND frozen graph into memory")
        #set GPU id before importing tensorflow!!!!!!!!!!!!!
        #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        #import tensorflow here
        #import tensorflow as tf
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.io.gfile.GFile(self.chkpt_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph, config=config)
        print(">  ====== Hand Inference graph loaded.")
        return detection_graph, sess


    # Actual detection .. generate scores and bounding boxes given an image
    def detect_objects(self, image_np, detection_graph, sess):
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name(
            'detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name(
            'num_detections:0')

        image_np_expanded = np.expand_dims(image_np, axis=0)

        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores,
                detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        return np.squeeze(boxes), np.squeeze(scores)

    def worker(self, input_q, output_q):
        #set GPU id before importing tensorflow!!!!!!!!!!!!!
        #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        #import tensorflow here
        #import tensorflow as tf
        print(">> loading frozen model for worker")
        detection_graph, sess = self.load_inference_graph()
        sess = tf.Session(graph=detection_graph, config=config)
        while True:
            #print("> ===== in worker loop, frame ", frame_processed)
            frame = self.input_q.get()
            im_height, im_width = frame.shape[:2]
            if (frame is not None):
                # actual detection
                boxes, scores = self.detect_objects(frame, detection_graph, sess)
                # draw bounding boxes
                #imutils.draw_box_on_image(self.num_hands_detect, self.score_thresh, scores, boxes, im_width, im_height, frame)
                box = imutils.getWindow(self.num_hands_detect, self.score_thresh, scores, boxes, im_width, im_height, frame)
                print("box = ", box)
                self.output_q.put(box)
                #frame_processed += 1
            else:
                self.output_q.put(box)
        sess.close()

    def initPool(self):
        self.pool = Pool(self.nworkers, self.worker,(self.input_q, self.output_q))

    def detect(self, frame):
        #frame = cv.flip(frame, 1)
        self.input_q.put(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        box = self.output_q.get()
        if (box is not None):
            return box#cv.cvtColor(output_frame, cv.COLOR_RGB2BGR)
        else:
            return box
            #self.pool.terminate()

def selectDetector(detector_type):
    cascade_xml = "detection_files/aGest.xml"
    dir = "detection_files/"
    model_path = "detection_files/inf_graphs/frozen_graph_viva_200k"
    #model_path = "detection_files/inf_graphs/ssd_mobilenet_coco_v2_41098steps"
    #model_path = "detection_files/inf_graphs/ssd_mobilenet_coco_v1_101ksteps"
    #model_path = "detection_files/inf_graphs/ssd_mobilenet_coco_v1_18ksteps"
    #model_path = "detection_files/inf_graphs/frozen_graph_orig_config79k"
    #model_path = 'detection_files/inf_graphs/ssd_mobilenet_v2_config_104ksteps'
    #model_path = 'detection_files/inf_graphs/frozen_graph4_config102k'
    #model_path = 'detection_files/inf_graphs/frozen_graph4_config151k'
    #model_path = 'detection_files/inf_graphs/frozen_graph4_config170k'
    #model_path = 'detection_files/inf_graphs/frozen_graph4_config200k'
    label_path = "detection_files/hand_label_map.pbtxt"
    if detector_type == 'cascade':
        detector = cascadeDetection(cascade_xml)
    elif detector_type == 'color':
        detector = colorDetection(dir)
    elif detector_type == 'movement':
        detector = movementDetection()
    elif detector_type == 'cnn_st':
        detector = cnnDetectionSingleThread(model_path,label_path)
    elif detector_type == 'cnn_mt':
        detector = cnnDetectionMultiThread(model_path,label_path)

    return detector
