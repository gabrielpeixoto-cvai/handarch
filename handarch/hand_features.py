#Extract features for static gestures recognition
import cv2 as cv
import numpy as np
from time import time
from sklearn.externals import joblib
import os
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import sys
from tqdm import tqdm

class HOGFeatures:
    def __init__(self):
        #image = cv.imread("test.jpg",0)
        self.winSize = (80,80)
        self.blockSize = (16,16)
        self.blockStride = (8,8)
        self.cellSize = (8,8)
        self.nbins = 9
        self.derivAperture = 1
        self.winSigma = 4.
        self.histogramNormType = 0
        self.L2HysThreshold = 2.0000000000000001e-01
        self.gammaCorrection = 1
        #nlevels = 10
        self.hog = cv.HOGDescriptor(self.winSize,self.blockSize,self.blockStride,self.cellSize,self.nbins,self.derivAperture,self.winSigma,self.histogramNormType,self.L2HysThreshold,self.gammaCorrection)#,nlevels)
        #
        self.winStride = (16,16)
        self.padding = (8,8)

    def dimensionReductionPCA(self, feature):
        #pca = cv.PCACompute(feature, mean=None, maxComponents=1000)
        #pca = cv.PCA(feature, mean=None, maxComponents=1000)
        t0 = time()
        mean, eigenvectors = cv.PCACompute(feature, mean=None, maxComponents=1000)
        print("PCA compute took ", time()-t0,"seconds to run")
        #reduced_features = pca.project(feature)
        t1 = time()
        reduced_features = cv.PCAProject(feature, mean, eigenvectors)
        print("PCA project took ", time()-t1,"seconds to run")
        return reduced_features


    def extractFeatures(self, image):
        im = cv.resize(image,(80, 80), interpolation = cv.INTER_CUBIC)
        fd = self.hog.compute(im,self.winStride,self.padding)
        return fd

class HOGPCAFeatures:
    def __init__(self):
        #image = cv.imread("test.jpg",0)
        self.winSize = (80,80)
        self.blockSize = (16,16)
        self.blockStride = (8,8)
        self.cellSize = (8,8)
        self.nbins = 9
        self.derivAperture = 1
        self.winSigma = 4.
        self.histogramNormType = 0
        self.L2HysThreshold = 2.0000000000000001e-01
        self.gammaCorrection = 1
        #nlevels = 10
        self.hog = cv.HOGDescriptor(self.winSize,self.blockSize,self.blockStride,self.cellSize,self.nbins,self.derivAperture,self.winSigma,self.histogramNormType,self.L2HysThreshold,self.gammaCorrection)#,nlevels)
        #
        self.winStride = (16,16)
        self.padding = (8,8)
        self.pcapath = "/media/1tb/datasets/libras_configurations/chroma_videos/features/"
        self.mean_name = "pcahog_skin.mean"
        self.eigenvectors_names = "pcahog_skin.eigv"
        self.mean_path = os.path.join(self.pcapath, self.mean_name)
        self.eigenvectors_path = os.path.join(self.pcapath, self.eigenvectors_names)
        self.reduced_features_name = "reduced_features1000.feat"
        self.features_path = os.path.join(self.pcapath, self.reduced_features_name)
        #self.mean = joblib.load(self.mean_path)
        #self.eigenvector = joblib.load(self.eigenvectors_path)

    def computePCA(self, feature):
        #pca = cv.PCACompute(feature, mean=None, maxComponents=1000)
        #pca = cv.PCA(feature, mean=None, maxComponents=1000)
        t0 = time()
        mean, eigenvectors = cv.PCACompute(feature, mean=None, maxComponents=2000)
        print("PCA compute took ", time()-t0,"seconds to run")
        #reduced_features = pca.project(feature)
        print("Saving Means and eigenvectors")
        #joblib.dump(mean, self.mean_path, compress=True)
        #joblib.dump(eigenvectors, self.eigenvectors_path, compress=True)
        return mean, eigenvectors

    def projectPCA(self, mean, eigenvectors, feature):
        t1 = time()
        reduced_features = cv.PCAProject(feature, mean, eigenvectors)
        print("PCA project took ", time()-t1,"seconds to run")
        return reduced_features

    def extractFeatures(self, image):
        im = cv.resize(image,(80, 80), interpolation = cv.INTER_CUBIC)
        fd = self.hog.compute(im,self.winStride,self.padding)
        return fd

    def extract(self, image):
        #print(self.mean.shape)
        #print(self.eigenvector.shape)
        features = self.extractFeatures(image)
        print(features.shape)
        features = np.transpose(features)
        print(features.shape)
        reduced_features = self.projectPCA(self.mean, self.eigenvector, features)
        return reduced_features


class HUFeatures:
    def __init__(self):
        self.initial =0

    def getContours(self, img):
        contours,hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

        #contours_retain=[]
        #for cnt in contours:
        #    if  cv.contourArea(cnt)>500:
        #        contours_retain.append(cnt)
        c = max(contours, key = cv.contourArea)

        #cv.drawContours(img,contours_retain,-1,(255,0,255),3)

        return c

    def extractFeatures(self, img):

        image = cv.resize(img,(200, 200), interpolation = cv.INTER_CUBIC)

        c = self.getContours(image)
        #for cnt in contours_retain:
        #    print(cv.HuMoments(cv.moments(cnt)).flatten())
        ft = cv.HuMoments(cv.moments(c)).flatten()
        return ft

class GaborFeatures:
    def __init__(self):
        self.x = 0
        self.filters = self.build_filters()

    # define gabor filter bank with different orientations and at different scales
    def build_filters(self):
        filters = []
        ksize = 9
        #define the range for theta and nu
        for theta in np.arange(0, np.pi, np.pi / 8):
            for nu in np.arange(0, 6*np.pi/4 , np.pi / 4):
                kern = cv.getGaborKernel((ksize, ksize), 1.0, theta, nu, 0.5, 0, ktype=cv.CV_32F)
                kern /= 1.5*kern.sum()
                filters.append(kern)
        return filters

        #function to convolve the image with the filters
    def process(self, img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv.filter2D(img, cv.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum

    def extractFeatures(self, img):
        #instantiating the filters
        img = cv.resize(img,(128, 128), interpolation = cv.INTER_CUBIC)
        #filters = self.build_filters()
        f = np.array(self.filters, dtype='int64')
        #initializing the feature vector
        feat = []
        #calculating the local energy for each convolved image
        for j in range(40):
            res = self.process(img, f[j])
            #print(res.shape)
            temp = np.dtype('int64').type(0)
            for p in range(128):
                for q in range(128):
                    temp = temp + res[p][q]*res[p][q]
            #print(temp)
            feat.append(temp)

        #calculating the mean amplitude for each convolved image
        for j in range(40):
            res = self.process(img, f[j])
            #print(res.shape)
            temp = np.dtype('int64').type(0)
            for p in range(128):
                for q in range(128):
                    temp = temp + abs(res[p][q])
            #print(temp)
            feat.append(temp)

        feat = np.array(feat, dtype='int64')
        return feat
#feat matrix is the feature vector for the image


class SIFTFeatures:
    def __init__(self):
        # Create feature extraction and keypoint detector objects
        #old opencv 2
        #self.fea_det = cv.FeatureDetector_create("SIFT")
        #new opencv 3
        self.sift = cv.xfeatures2d.SIFT_create()
        #old opencv
        #self.star = cv.FeatureDetector_create("STAR")
        #self.brief = cv.DescriptorExtractor_create("BRIEF")
        #new opencv 3.4+
        #self.star = cv.xfeatures2d.StarDetector_create()
        self.surf = cv.xfeatures2d.SURF_create()
        self.brief = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes = 64, use_orientation = True)

        #old opencv 2
        #self.des_ext = cv.DescriptorExtractor_create("SIFT")
        self.des_list = []
        self.image_paths = []
        self.size =0
        self.none_images = []

    def computeDescriptor(self, img, image_path):
        # opencv 2 api
        #kpts = self.fea_det.detect(img)
        #kpts, des = self.des_ext.compute(img, kpts)
        img = cv.resize(img,(128, 128), interpolation = cv.INTER_CUBIC)
        (kpts, des) = self.sift.detectAndCompute(img, None)
        #kp = self.star.detect(img,None)
        #kp, des = self.surf.detectAndCompute(img, None)
        #(kp, des) = self.brief.compute(img, kp)
        

        #print(des.shape)
        if des is None:
            #print(image_path)
            self.none_images.append(image_path)
            return -1
        else:
            #print(des.shape)
            des = des.astype(np.float32)
            #print(des.dtype)
            self.des_list.append((image_path, des))
            self.image_paths.append(image_path)
            #print(image_path)
            #print(des.shape)
            self.size += len(des)
            return 0
        #print(self.des_list)

    def extractFeatures(self):
        # Stack all the descriptors vertically in a numpy array
        descriptors = self.des_list[0][1]
        #print(len(self.des_list[0][1]))
        if len(self.none_images) > 0:
            print("there were images with errors")
            print(self.none_images)
            print(len(self.none_images))
        print("concatenating descriptors")
        count = 0
        t0 = time()
        for image_path, descriptor in tqdm(self.des_list[1:]):
            count+=1
            descriptors = np.vstack((descriptors, descriptor))
            #print(count)
            #print(descriptors.shape)
        print("concatenation done in ", time()-t0, "seconds")

        t0=time()

        #print(descriptors.shape)
        print("preforming k-means clustering")
        # Perform k-means clustering
        k = 200
        voc, variance = kmeans(descriptors, k, 1)

        print("Kmeans done in ", time()-t0, "seconds")

        # Calculate the histogram of features
        im_features = np.zeros((len(self.image_paths), k), "float32")
        for i in tqdm(range(len(self.image_paths))):
            words, distance = vq(self.des_list[i][1],voc)
            for w in words:
                im_features[i][w] += 1

        # Perform Tf-Idf vectorization
        nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
        idf = np.array(np.log((1.0*len(self.image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

        # Scaling the words
        stdSlr = StandardScaler().fit(im_features)
        im_features = stdSlr.transform(im_features)
        return im_features


class SURFFeatures:
    def __init__(self):
        # Create feature extraction and keypoint detector objects
        #old opencv 2
        #self.fea_det = cv.FeatureDetector_create("SIFT")
        #new opencv 3
        self.surf = cv.xfeatures2d.SURF_create()
        #old opencv 2
        #self.des_ext = cv.DescriptorExtractor_create("SIFT")
        self.des_list = []
        self.image_paths = []
        self.size = 0
        self.none_images = []

    def computeDescriptor(self, img, image_path):
        # opencv 2 api
        #kpts = self.fea_det.detect(img)
        #kpts, des = self.des_ext.compute(img, kpts)
        img = cv.resize(img,(128, 128), interpolation = cv.INTER_CUBIC)
        (kpts, des) = self.surf.detectAndCompute(img, None)
        #print(des.shape)
        if des is None:
            #print(image_path)
            self.none_images.append(image_path)
            return -1
        else:
            self.des_list.append((image_path, des))
            self.image_paths.append(image_path)
            #print(image_path)
            self.size += len(des)
            return 0
        #print(self.des_list)

    def extractFeatures(self):
        # Stack all the descriptors vertically in a numpy array
        descriptors = self.des_list[0][1]
        #print(len(self.des_list[0][1]))
        if len(self.none_images) > 0:
            print("there were images with errors")
            #print(self.none_images)
        print("concatenating descriptors")
        count = 0
        t0 = time()
        for image_path, descriptor in tqdm(self.des_list[1:]):
            count+=1
            #print(self.size)
            #print(image_path)
            #print(np.array(descriptor).shape)
            #print(descriptors.shape)
            descriptors = np.vstack((descriptors, descriptor))
            #print(count)
        print("concatenation done in ", time()-t0, "seconds")

        t0=time()

        #print(descriptors.shape)
        # Perform k-means clustering
        k = 200
        print("preforming k-means clustering with ", k, "clusters")
        voc, variance = kmeans(descriptors, k, 1)

        print("Kmeans done in ", time()-t0, "seconds")

        # Calculate the histogram of features
        im_features = np.zeros((len(self.image_paths), k), "float32")
        for i in tqdm(range(len(self.image_paths))):
            words, distance = vq(self.des_list[i][1],voc)
            for w in words:
                im_features[i][w] += 1

        # Perform Tf-Idf vectorization
        nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
        idf = np.array(np.log((1.0*len(self.image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

        # Scaling the words
        stdSlr = StandardScaler().fit(im_features)
        im_features = stdSlr.transform(im_features)
        print(self.none_images)
        return im_features
