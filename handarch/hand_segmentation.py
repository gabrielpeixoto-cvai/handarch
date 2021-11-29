import cv2 as cv
import re
import numpy as np
from time import time
import os
import pickle
import image_utils as imutils

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import logging

from skimage.exposure import rescale_intensity

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float


model_dir = "model/"
#model_dir = "."
covs_path = model_dir+"conv_em.dat"
means_path = model_dir+"mean_em.dat"

class EMsegmentation:
    def __init__(self):
        self.means = pickle.load( open( means_path, "rb" ) )
        self.covs = pickle.load( open( covs_path, "rb" ) )
        #self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        self.kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
        self.no_of_clusters = 2

    def segment(self, img):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        output = hsv.copy()
        colors = np.array([[255, 255, 255], [0, 0, 0]])
        x, y, z = img.shape
        distance = [0] * self.no_of_clusters
        t0 = time()
        for i in range(x):
            for j in range(y):
                for k in range(self.no_of_clusters):
                    diff = hsv[i, j] - self.means[k]
                    distance[k] = abs(np.dot(np.dot(diff, self.covs[k]), diff.T))
                output[i][j] = colors[distance.index(max(distance))]
        #print("Segmentation done in %0.3fs" % (time() - t0))
        output = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
        output = cv.morphologyEx(output, cv.MORPH_OPEN, self.kernel)
        output = cv.morphologyEx(output, cv.MORPH_CLOSE, self.kernel)
        output = cv.dilate(output,self.kernel,iterations = 1)
        roi = cv.cvtColor(output,cv.COLOR_GRAY2BGR)
        return roi

class BackPropagationSegmentation:
    def __init__(self):
        #self.disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        self.kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
        #initialize the kernel to use in the morphological operations
        self.kernel = np.ones((5,5),np.uint8)


    def segment(self, roi):
        #get roi dimensions
        roi_height, roi_width, roi_channels = roi.shape

        #print(roi_width,roi_height)

        #get center point
        centerx = int(roi_width/2)
        centery = int(roi_height/2)

        #print(centerx,centery)

        #get a square of 50x50 centered in the center point
        #square roi points
        sqrt_x = centerx - 25
        sqrt_y = centery - 25
        sqrt_w = 50
        sqrt_h = 50

        #get the square roi from the image
        sqrt_roi = roi[sqrt_y:sqrt_y+sqrt_h, sqrt_x:sqrt_x+sqrt_w]
        #cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

        #cv.imshow("square_roi", sqrt_roi)

        hsv = cv.cvtColor(sqrt_roi,cv.COLOR_BGR2HSV)
        hsvt = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
        # calculating object histogram
        roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
        # normalize histogram and apply backprojection
        cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
        disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        kernel = np.ones((5,5),np.uint8)
        dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
        cv.filter2D(dst,-1,disc,dst)
        # threshold and binary AND
        ret,thresh = cv.threshold(dst,50,255,0)
        thresh = cv.merge((thresh,thresh,thresh))
        res = cv.bitwise_and(roi,thresh)
        #res = np.vstack((target,thresh,res))
        #res = cv.morphologyEx(res, cv.MORPH_OPEN, kernel)
        #res = cv.morphologyEx(res, cv.MORPH_CLOSE, kernel)
        #res = cv.morphologyEx(res, cv.MORPH_CLOSE, kernel)
        #drawCross(res,[centerx,centery],(255, 0, 255), 2)
        #cv.rectangle(res,(sqrt_x,sqrt_y),(sqrt_x+sqrt_w,sqrt_y+sqrt_h),(0,0,255),2)
        thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
        thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, self.kernel)
        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, self.kernel)
        thresh = cv.dilate(thresh,self.kernel,iterations = 1)
        thresh = cv.cvtColor(thresh,cv.COLOR_GRAY2BGR)
        thresh = cv.blur(thresh,(5,5))
        return thresh

class StaticColorSegmentation:
    def __init__(self):
        # Constants for finding range of skin color in YCrCb
        self.min_YCrCb = np.array([0,133,77],np.uint8)
        self.max_YCrCb = np.array([255,173,127],np.uint8)
        self.kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))

    def segment(self, roi):
        t0 =  time()
        # Convert image to YCrCb
        imageYCrCb = cv.cvtColor(roi,cv.COLOR_BGR2YCR_CB)

        # Find region with skin tone in YCrCb image
        skinRegion = cv.inRange(imageYCrCb,self.min_YCrCb,self.max_YCrCb)

        #area,roi,x,y,w,h = imutils.FindBiggestContour(skinRegion)

        t1 = (time())-t0

        fps = 1/t1

        skinRegion = cv.morphologyEx(skinRegion, cv.MORPH_OPEN, self.kernel)
        skinRegion = cv.morphologyEx(skinRegion, cv.MORPH_CLOSE, self.kernel)

        #skinRegion = cv.blur(skinRegion,(3,3))

        roi = cv.cvtColor(skinRegion,cv.COLOR_GRAY2BGR)

        #print("roi processed in: ", t1, "s")
        #print("Theoretical fps: ", fps)

        return roi

class BGSegmentation:
    def __init__(self):
        self.bgSubThreshold = 50
        self.learningRate = 0
        self.kernel = np.ones((3, 3), np.uint8)
        self.bgModel = cv.createBackgroundSubtractorMOG2(0, self.bgSubThreshold)

    def segment(self, frame):
        fgmask = self.bgModel.apply(frame,learningRate=self.learningRate)
        fgmask = cv.erode(fgmask, self.kernel, iterations=1)
        res = cv.bitwise_and(frame, frame, mask=fgmask)
        return res

class TreeSegmentation:
    def __init__(self):
        self.model_path =  "models/tree_skin.mod"
        self.clf = joblib.load(self.model_path)

    def ReadData(self):
        #Data in format [B G R Label] from
        data = np.genfromtxt('skin_nskin.txt', dtype=np.int32)

        labels= data[:,3]
        data= data[:,0:3]

        return data, labels

    def BGR2HSV(self, bgr):
        bgr= np.reshape(bgr,(bgr.shape[0],1,3))
        hsv= cv.cvtColor(np.uint8(bgr), cv.COLOR_BGR2HSV)
        hsv= np.reshape(hsv,(hsv.shape[0],3))

        return hsv

    def TrainTree(self, data, labels, flUseHSVColorspace):
        if(flUseHSVColorspace):
            data= BGR2HSV(data)

        trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.20, random_state=42)

        #print(trainData.shape)
        #print(trainLabels.shape)
        #print(testData.shape)
        #print(testLabels.shape)

        clf = tree.DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(trainData, trainLabels)
        print(clf.feature_importances_)
        print(clf.score(testData, testLabels))
        if not os.path.isdir(os.path.split(self.model_path)[0]):
            os.makedirs(os.path.split(self.model_path)[0])
        joblib.dump(clf, self.model_path)

        return clf

    def apply(self, img, flUseHSVColorspace=True):
        data= np.reshape(img,(img.shape[0]*img.shape[1],3))
        print(data.shape)

        if(flUseHSVColorspace):
            data= self.BGR2HSV(data)

        t0 = time()

        predictedLabels= self.clf.predict(data)

        imgLabels= np.reshape(predictedLabels,(img.shape[0],img.shape[1],1))

        res = ((-(imgLabels-1)+1)*255)

        print("Tree classifier segmentation done in ", time()-t0, " seconds")

        return res

    def segment(self, image):
        h, w = image.shape[:2]
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
        res = self.apply(image)
        res = rescale_intensity(res, out_range=(0, 255)).astype("uint8")
        res = cv.resize(res,(w*2,h*2), interpolation = cv.INTER_CUBIC)
        res = cv.morphologyEx(res, cv.MORPH_OPEN, kernel)
        res = cv.morphologyEx(res, cv.MORPH_CLOSE, kernel)
        res = cv.resize(res,(w,h), interpolation = cv.INTER_CUBIC)
        res = cv.cvtColor(res,cv.COLOR_GRAY2BGR)
        return res

class BayesSegmentation:
    def __init__(self):
        self.model_path =  "models/bayes_skin_sfa.mod"
        self.clf = joblib.load(self.model_path)
        self.kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
        self.kernel2 = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
        self.kernel3 = cv.getStructuringElement(cv.MORPH_RECT,(7,7))
        self.kernel4 = cv.getStructuringElement(cv.MORPH_RECT,(9,9))

    def ReadData(self):
        #Data in format [B G R Label] from
        data = np.genfromtxt('skin_nskin.txt', dtype=np.int32)

        labels= data[:,3]
        data= data[:,0:3]

        return data, labels

    def BGR2HSV(self, bgr):
        bgr= np.reshape(bgr,(bgr.shape[0],1,3))
        hsv= cv.cvtColor(np.uint8(bgr), cv.COLOR_BGR2HSV)
        hsv= np.reshape(hsv,(hsv.shape[0],3))

        return hsv

    def TrainBayes(self, data, labels, flUseHSVColorspace=True):
        #self.model_path =  "models/bayes_skin_sfa.mod"
        if(flUseHSVColorspace):
            data= self.BGR2HSV(data)

        trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.20, random_state=42)

        clf = GaussianNB()
        clf = clf.fit(trainData, trainLabels)
        #print(clf.feature_importances_)
        print(clf.score(testData, testLabels))
        if not os.path.isdir(os.path.split(self.model_path)[0]):
            os.makedirs(os.path.split(self.model_path)[0])
        joblib.dump(clf, self.model_path)

        return clf

    def apply(self, img, flUseHSVColorspace=True):
        data= np.reshape(img,(img.shape[0]*img.shape[1],3))
        #print(data.shape)

        if(flUseHSVColorspace):
            data= self.BGR2HSV(data)

        t0 = time()

        predictedLabels= self.clf.predict_proba(data)

        imgLabels= np.reshape(predictedLabels[:,0],(img.shape[0],img.shape[1],1))

        res = imgLabels*255

        #print("Bayes classifier segmentation done in ", time()-t0, " seconds")

        return res

    def segment(self, image, thresh=False):
        h, w = image.shape[:2]
        t0 = time()
        res = self.apply(image)
        res = rescale_intensity(res, out_range=(0, 255)).astype("uint8")
        res = cv.morphologyEx(res, cv.MORPH_CLOSE, self.kernel2)
        res = cv.resize(res,(w,h), interpolation = cv.INTER_CUBIC)
        if thresh == True:
            ret, res = cv.threshold(res,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        #res = cv.equalizeHist(res)
        #res = cv.erode(res,self.kernel,iterations = 1)
        res = cv.dilate(res,self.kernel,iterations = 2)
        res = cv.morphologyEx(res, cv.MORPH_CLOSE, self.kernel2)
        res = cv.cvtColor(res,cv.COLOR_GRAY2BGR)
        print("bayes segmentation done in: ", time()-t0, "s")
        return res

class MLPSegmentation:
    def __init__(self):
        self.model_path =  "models/mlp_skin.mod"
        self.clf = joblib.load(self.model_path)
        self.kernel3 = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
        self.kernel5 = cv.getStructuringElement(cv.MORPH_RECT,(5,5))

    def ReadData(self):
        #Data in format [B G R Label] from
        data = np.genfromtxt('skin_nskin.txt', dtype=np.int32)

        labels= data[:,3]
        data= data[:,0:3]

        return data, labels

    def BGR2HSV(self, bgr):
        bgr= np.reshape(bgr,(bgr.shape[0],1,3))
        hsv= cv.cvtColor(np.uint8(bgr), cv.COLOR_BGR2HSV)
        hsv= np.reshape(hsv,(hsv.shape[0],3))

        return hsv

    def TrainMLP(self, data, labels, flUseHSVColorspace):
        if(flUseHSVColorspace):
            data= BGR2HSV(data)

        trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.20, random_state=42)

        #print(trainData.shape)
        #print(trainLabels.shape)
        #print(testData.shape)
        #print(testLabels.shape)

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100), random_state=1)
        clf = clf.fit(trainData, trainLabels)
        print(clf.feature_importances_)
        print(clf.score(testData, testLabels))
        if not os.path.isdir(os.path.split(self.model_path)[0]):
            os.makedirs(os.path.split(self.model_path)[0])
        joblib.dump(clf, self.model_path)

        return clf

    def apply(self, img, flUseHSVColorspace=True):
        data= np.reshape(img,(img.shape[0]*img.shape[1],3))
        #print(data.shape)

        if(flUseHSVColorspace):
            data= self.BGR2HSV(data)

        t0 = time()

        predictedLabels= self.clf.predict_proba(data)

        if 1:
            imgLabels= np.reshape(predictedLabels[:,0],(img.shape[0],img.shape[1],1))
            res = imgLabels*255
        else:
            imgLabels= np.reshape(predictedLabels,(img.shape[0],img.shape[1],1))
            res = ((-(imgLabels-1)+1)*255)

        #print("MLP classifier segmentation done in ", time()-t0, " seconds")

        return res

    def segment(self, image):
        h, w = image.shape[:2]
        t0 = time()
        res = self.apply(image)
        res = rescale_intensity(res, out_range=(0, 255)).astype("uint8")
        #res = cv.resize(res,(w*2,h*2), interpolation = cv.INTER_CUBIC)
        #res = cv.morphologyEx(res, cv.MORPH_OPEN, self.kernel3)
        res = cv.morphologyEx(res, cv.MORPH_CLOSE, self.kernel5)
        res = cv.resize(res,(w,h), interpolation = cv.INTER_CUBIC)
        ret, res = cv.threshold(res,10,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        res = cv.cvtColor(res,cv.COLOR_GRAY2BGR)
        print("MLP segmentation done in: ", time()-t0, "s")
        return res

class SVMSegmentation:
    def __init__(self):
        self.model_path =  "models/svm_skin.mod"
        self.clf = joblib.load(self.model_path)

    def ReadData(self):
        #Data in format [B G R Label] from
        data = np.genfromtxt('skin_nskin.txt', dtype=np.int32)

        labels= data[:,3]
        data= data[:,0:3]

        return data, labels

    def BGR2HSV(self, bgr):
        bgr= np.reshape(bgr,(bgr.shape[0],1,3))
        hsv= cv.cvtColor(np.uint8(bgr), cv.COLOR_BGR2HSV)
        hsv= np.reshape(hsv,(hsv.shape[0],3))

        return hsv

    def TrainSVM(self, data, labels, flUseHSVColorspace=True):
        if(flUseHSVColorspace):
            data= BGR2HSV(data)

        trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.20, random_state=42)

        #print(trainData.shape)
        #print(trainLabels.shape)
        #print(testData.shape)
        #print(testLabels.shape)

        clf = SVC(kernel='linear', class_weight='balanced', probability=False, verbose=True)
        clf = clf.fit(trainData, trainLabels)
        print(clf.feature_importances_)
        print(clf.score(testData, testLabels))
        if not os.path.isdir(os.path.split(self.model_path)[0]):
            os.makedirs(os.path.split(self.model_path)[0])
        joblib.dump(clf, self.model_path)

        return clf

    def apply(self, img, flUseHSVColorspace=True):
        data= np.reshape(img,(img.shape[0]*img.shape[1],3))
        #print(data.shape)

        if(flUseHSVColorspace):
            data= self.BGR2HSV(data)

        t0 = time()

        if 0:
            predictedLabels= clf.predict_proba(data)
            #print(predictedLabels.shape)
            #print(predictedLabels[:,1])
            imgLabels= np.reshape(predictedLabels[:,0],(img.shape[0],img.shape[1],1))

            res = imgLabels*255
        else:
            predictedLabels= self.clf.predict(data)
            #print(predictedLabels.shape)
            imgLabels= np.reshape(predictedLabels,(img.shape[0],img.shape[1],1))
            res = ((-(imgLabels-1)+1)*255)
        #print("SVM segmentation done in ", time()-t0, " seconds")

        return res

    def segment(self, image):
        t0 = time()
        res = self.apply(image)
        res = rescale_intensity(res, out_range=(0, 255)).astype("uint8")
        res = cv.cvtColor(res,cv.COLOR_GRAY2BGR)
        print("SVM segmentation done in ", time()-t0, " seconds")
        return res

class BayesSlicSegmentation:
    def __init__(self):
        self.bayes_segment = BayesSegmentation()

    def superpixelClassifer(self, image, mask):
        (R, G, B) = cv.split(image.astype("float"))
        R = np.ma.masked_array(R, mask=mask)
        G = np.ma.masked_array(G, mask=mask)
        B = np.ma.masked_array(B, mask=mask)

        r = int(R.mean())
        g = int(G.mean())
        b = int(B.mean())

        pixel = np.array([[r],[g],[b]],np.uint8)
        pixel = pixel.reshape((1,1,3))

        res = self.bayes_segment.apply(pixel)

        return res.reshape((1))


    def slicClassifier(self,image):
        orig = image.copy()
        vis = np.zeros(orig.shape[:2], dtype="float")
        segments = slic(img_as_float(image),max_iter=10,n_segments=100,slic_zero=True)
        for v in np.unique(segments):
            # construct a mask for the segment so we can compute image
            # statistics for *only* the masked region
            mask = np.ones(image.shape[:2])
            mask[segments == v] = 0
            # compute the superpixel colorfulness, then update the
            # visualization array
            #C = segment_colorfulness(orig, ma
            C = self.superpixelClassifer(orig,mask)
            vis[segments == v] = C
        # scale the visualization image from an unrestricted floating point
        # to unsigned 8-bit integer array so we can use it with OpenCV and
        # display it to our screen
        vis = rescale_intensity(vis, out_range=(0, 255)).astype("uint8")
        # overlay the superpixel colorfulness visualization on the original
        # image
        alpha = 0.6
        overlay = np.dstack([vis] * 3)
        output = orig.copy()
        cv.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        vis = cv.cvtColor(vis,cv.COLOR_GRAY2BGR)
        return vis

    def segment(self, image):
        t0 = time()
        ret = self.slicClassifier(image)
        print("bayes+SLIC segmentation done in ",time()-t0,"s")
        return ret


class MlpSlicSegmentation:
    def __init__(self):
        self.mlp_segment = MLPSegmentation()

    def superpixelClassifer(self, image, mask):
        (R, G, B) = cv.split(image.astype("float"))
        R = np.ma.masked_array(R, mask=mask)
        G = np.ma.masked_array(G, mask=mask)
        B = np.ma.masked_array(B, mask=mask)

        r = int(R.mean())
        g = int(G.mean())
        b = int(B.mean())

        pixel = np.array([[r],[g],[b]],np.uint8)
        pixel = pixel.reshape((1,1,3))

        res = self.mlp_segment.apply(pixel)

        return res.reshape((1))


    def slicClassifier(self,image):
        orig = image.copy()
        vis = np.zeros(orig.shape[:2], dtype="float")
        segments = slic(img_as_float(image),max_iter=10,n_segments=100,slic_zero=True)
        for v in np.unique(segments):
            # construct a mask for the segment so we can compute image
            # statistics for *only* the masked region
            mask = np.ones(image.shape[:2])
            mask[segments == v] = 0
            # compute the superpixel colorfulness, then update the
            # visualization array
            #C = segment_colorfulness(orig, ma
            C = self.superpixelClassifer(orig,mask)
            vis[segments == v] = C
        # scale the visualization image from an unrestricted floating point
        # to unsigned 8-bit integer array so we can use it with OpenCV and
        # display it to our screen
        vis = rescale_intensity(vis, out_range=(0, 255)).astype("uint8")
        # overlay the superpixel colorfulness visualization on the original
        # image
        alpha = 0.6
        overlay = np.dstack([vis] * 3)
        output = orig.copy()
        cv.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        vis = cv.cvtColor(vis,cv.COLOR_GRAY2BGR)
        return vis

    def segment(self, image):
        t0 = time()
        ret = self.slicClassifier(image)
        print("MLP+SLIC segmentation done in ",time()-t0,"s")
        return ret



def selectSegmentation(segmentation_type):
    if segmentation_type == 'em_segmentation':
        segmentation = EMsegmentation()
    elif segmentation_type == 'static_segmentation':
        segmentation = StaticColorSegmentation()
    elif segmentation_type == 'hist_backpropagation':
        segmentation = BackPropagationSegmentation()
    elif segmentation_type == 'bg_segmentation':
        segmentation = BGSegmentation()
    elif segmentation_type == 'tree_segmentation':
        segmentation = TreeSegmentation()
    elif segmentation_type == 'bayes_segmentation':
        segmentation = BayesSegmentation()
    elif segmentation_type == 'mlp_segmentation':
        segmentation = MLPSegmentation()
    elif segmentation_type == 'svm_segmentation':
        segmentation = SVMSegmentation()
    elif segmentation_type == 'bayes_slic_segmentation':
        segmentation = BayesSlicSegmentation()
    elif segmentation_type == 'mlp_slic_segmentation':
        segmentation = MlpSlicSegmentation()
    elif segmentation_type == 'segmentation':
        segmentation = TreeSegmentation()
    return segmentation
