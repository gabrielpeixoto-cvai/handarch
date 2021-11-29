import cv2 as cv
import sys
import math
import numpy as np
from time import time
import image_utils as imutils
import Condensation as cons

#OpenCV default trackers begin

class boostingTracking:

    def __init__(self, frame, bbox):
        self.tracker = cv.TrackerBoosting_create()
        self.tracker.init(frame, bbox)

    def track(self, frame):
        ok, bbox = self.tracker.update(frame)
        return ok, bbox

class milTracking:

    def __init__(self, frame, bbox):
        self.tracker = cv.TrackerMIL_create()
        self.tracker.init(frame, bbox)

    def track(self, frame):
        ok, bbox = self.tracker.update(frame)
        return ok, bbox

class kcfTracking:

    def __init__(self, frame, bbox):
        self.tracker = cv.TrackerKCF_create()
        self.tracker.init(frame, bbox)

    def track(self, frame):
        ok, bbox = self.tracker.update(frame)
        return ok, bbox

class tldTracking:

    def __init__(self, frame, bbox):
        self.tracker = cv.TrackerTLD_create()
        self.tracker.init(frame, bbox)

    def track(self, frame):
        ok, bbox = self.tracker.update(frame)
        return ok, bbox

class medianFlowTracking:

    def __init__(self, frame, bbox):
        self.tracker = cv.TrackerMedianFlow_create()
        self.tracker.init(frame, bbox)

    def track(self, frame):
        ok, bbox = self.tracker.update(frame)
        return ok, bbox

class goturnTracking:

    def __init__(self, frame, bbox):
        self.tracker = cv.TrackerGOTURN_create()
        self.tracker.init(frame, bbox)

    def track(self, frame):
        ok, bbox = self.tracker.update(frame)
        return ok, bbox

#OpenCV default trackers end


#Meanshift tracking

class MeanShiftTracking:

    def __init__(self, frame, bbox):
        self.x,self.y,self.w,self.h = bbox
        self.track_window = bbox
        self.roi = imutils.getROI(frame,self.x,self.y,self.w,self.h)
        self.hsv_roi =  cv.cvtColor(self.roi, cv.COLOR_BGR2HSV)
        self.mask = cv.inRange(self.hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        self.roi_hist = cv.calcHist([self.hsv_roi],[0],self.mask,[180],[0,180])
        cv.normalize(self.roi_hist,self.roi_hist,0,255,cv.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        self.term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

    def restart(self, frame, bbox):
        self.x,self.y,self.w,self.h = bbox
        self.track_window = bbox
        self.roi = imutils.getROI(frame,self.x,self.y,self.w,self.h)
        self.hsv_roi =  cv.cvtColor(self.roi, cv.COLOR_BGR2HSV)
        self.mask = cv.inRange(self.hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        self.roi_hist = cv.calcHist([self.hsv_roi],[0],self.mask,[180],[0,180])
        cv.normalize(self.roi_hist,self.roi_hist,0,255,cv.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        self.term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

    def track(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],self.roi_hist,[0,180],1)
        # apply meanshift to get the new location
        #print(self.track_window)
        ret, self.track_window = cv.meanShift(dst, tuple(self.track_window), self.term_crit)
        # Draw it on image
        #x,y,w,h = track_window
        return ret, self.track_window

# camshfit tracking

class CAMShiftTracking:

    def __init__(self, frame, bbox):
        self.x,self.y,self.w,self.h = bbox
        self.track_window = bbox
        self.roi = imutils.getROI(frame,self.x,self.y,self.w,self.h)
        self.hsv_roi =  cv.cvtColor(self.roi, cv.COLOR_BGR2HSV)
        self.mask = cv.inRange(self.hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        self.roi_hist = cv.calcHist([self.hsv_roi],[0],self.mask,[180],[0,180])
        cv.normalize(self.roi_hist,self.roi_hist,0,255,cv.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        self.term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

    def restart(self,frame,bbox):
        self.x,self.y,self.w,self.h = bbox
        self.track_window = bbox
        self.roi = imutils.getROI(frame,self.x,self.y,self.w,self.h)
        self.hsv_roi =  cv.cvtColor(self.roi, cv.COLOR_BGR2HSV)
        self.mask = cv.inRange(self.hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        self.roi_hist = cv.calcHist([self.hsv_roi],[0],self.mask,[180],[0,180])
        cv.normalize(self.roi_hist,self.roi_hist,0,255,cv.NORM_MINMAX)
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        self.term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

    def track(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],self.roi_hist,[0,180],1)
        # apply meanshift to get the new location
        #print(self.track_window)
        ret, self.track_window = cv.CamShift(dst, tuple(self.track_window), self.term_crit)
        #ret = ((x,y),(w,h),angle)
        #print(ret)
        return ret, self.track_window

# optical flow tracking using shi-tomasi corner features

class OpticalFlowTracking:
    def __init__(self, frame, bbox):
        self.x,self.y,self.w,self.h = bbox
        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (40,40),
                          maxLevel = 3,
                          criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        self.old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.mask = imutils.getMask(self.old_gray,self.x,self.y,self.w,self.h)
        self.p0 = cv.goodFeaturesToTrack(self.old_gray, mask = self.mask, **self.feature_params)

    def track(self, frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask2 = np.zeros_like(frame)
        p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = self.p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            #mask2 = cv2.line(mask2, (a,b),(c,d), color[i].tolist(), 2)
            #frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv.add(frame,mask2)

        x,y,w,h = cv.boundingRect(p1)
        mask = imutils.getMask(frame_gray,x,y,w,h)
        self.p0 = cv.goodFeaturesToTrack(frame_gray, mask = mask, **self.feature_params)
        track_window = x,y,w,h
        self.old_gray = frame_gray.copy()
        self.p0 = good_new.reshape(-1,1,2)
        return err,track_window

#optical flow tracking using ORB features

class OpticalFlowTrackingORB:
    def __init__(self, frame, bbox):
        self.x,self.y,self.w,self.h = bbox
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (40,40),
                          maxLevel = 3,
                          criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        self.old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        self.roi = imutils.getROI(self.old_gray,self.x,self.y,self.w,self.h)
        # Initiate STAR detector
        self.orb = cv.ORB_create()

        # find the keypoints with ORB
        self.kp = self.orb.detect(self.roi,None)

        # compute the descriptors with ORB
        self.kp, self.des = self.orb.compute(self.roi, self.kp)
        self.p0=[]
        for keypoint in self.kp:
            #print(keypoint.pt)
            self.pty=keypoint.pt[0]+self.y
            self.ptx=keypoint.pt[1]+self.x
            self.p0.append([[self.ptx, self.pty]])
        self.p0=np.asfarray(self.p0, dtype=np.float32)

    def track(self, frame):
        print(len(self.p0))
        print("+++++++++")
        if(len(self.p0)==0):
            err = 1,1,1,1
            return 1,err
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask2 = np.zeros_like(frame)
        p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = self.p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
        img = cv.add(frame,mask2)

        x,y,w,h = cv.boundingRect(p1)
        print(x,y,w,h)
        print("=============")
        if (x>5 and y>5 and w>5 and h>5):
            roi = imutils.getROI(frame,x,y,w,h)
            # find the keypoints with ORB
            kp = self.orb.detect(roi,None)

            # compute the descriptors with ORB
            self.kp, self.des = self.orb.compute(roi, kp)
            self.p0=[]
            for keypoint in self.kp:
                pty=keypoint.pt[0]+y
                ptx=keypoint.pt[1]+x
                self.p0.append([[ptx, pty]])
            self.p0=np.asfarray(self.p0, dtype=np.float32)
        track_window = x,y,w,h
        self.old_gray = frame_gray.copy()
        self.p0 = good_new.reshape(-1,1,2)

        return err,track_window

#this class uses only camshift as sensor for the kalman filter

class KalmanTrackingCshift:
    def __init__(self, frame, bbox):
        self.kalman = cv.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],
                                             [0,1,0,0]],np.float32)#H

        self.kalman.transitionMatrix = np.array([[1,0,1,0],
                                            [0,1,0,1],
                                            [0,0,1,0],
                                            [0,0,0,1]],np.float32)#F

        self.kalman.processNoiseCov = np.array([[1,0,0,0],
                                           [0,1,0,0],
                                           [0,0,1,0],
                                           [0,0,0,1]],np.float32) * 0.03#P
        self.measurement = np.array((2,1), np.float32)
        self.prediction = np.zeros((2,1), np.float32)
        self.camshift = CAMShiftTracking(frame,bbox)
        self.prevx = 0
        self.prevy = 0

    def restart(self, frame, bbox):
        self.camshift = CAMShiftTracking(frame,bbox)

    def track(self, frame):
        ret, track_window = self.camshift.track(frame)
        #print("ret")
        #print(ret)
        # draw observation on image
        x,y,w,h = track_window;
        #frame = cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2);
        # extract centre of this observation as points

        pts = cv.boxPoints(ret)
        #print("Box Points:",pts)
        pts = np.int0(pts)
        #print(pts)
        #frame = cv.circle(frame,(pts[0][0],pts[0][1]), 5, (0,0,255), -1)
        #frame = cv.circle(frame,(pts[1][0],pts[1][1]), 5, (0,0,255), -1)
        #frame = cv.circle(frame,(pts[2][0],pts[2][1]), 5, (0,0,255), -1)
        #frame = cv.circle(frame,(pts[3][0],pts[3][1]), 5, (0,0,255), -1)
        # (cx, cy), radius = cv2.minEnclosingCircle(pts)

        # use to correct kalman filter

        self.kalman.correct(imutils.center(pts));

        # get new kalman filter prediction

        prediction = self.kalman.predict();
        #print("kf prediction:", prediction)

        #print("predictions")
        #print(prediction)

        x = prediction[0]-(0.5*w)

        y = prediction[1]-(0.5*h)

        #print("prevx and prev y =", self.prevx, self.prevy)
        #print("x and y=", x, y)
        #print("velocities: velx=",(x-self.prevx)/1,"vely=",(y-self.prevy)/1)

        w = prediction[0]+(0.5*w)-x

        h = prediction[1]+(0.5*h)-y

        self.prevx = x
        self.prevy = y


        prediction_box = x,y,w,h

        #print(prediction_box)

        #draw prediction in image
        #frame = cv.rectangle(frame, (prediction[0]-(0.5*w),prediction[1]-(0.5*h)), (prediction[0]+(0.5*w),prediction[1]+(0.5*h)), (0,255,0),2);

        ok = 1

        return ok, prediction_box


#This class uses CAMshift and optical flow as sensors for kalman filter

class KalmanTrackingCshiftOflow:
    def __init__(self, frame, bbox):
        self.kalman = cv.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],
                                             [0,1,0,0]],np.float32)

        self.kalman.transitionMatrix = np.array([[1,0,1,0],
                                            [0,1,0,1],
                                            [0,0,1,0],
                                            [0,0,0,1]],np.float32)

        self.kalman.processNoiseCov = np.array([[1,0,0,0],
                                           [0,1,0,0],
                                           [0,0,1,0],
                                           [0,0,0,1]],np.float32) * 0.03
        self.measurement = np.array((2,1), np.float32)
        self.prediction = np.zeros((2,1), np.float32)
        self.camshift = CAMShiftTracking(frame,bbox)
        self.opticalflow = OpticalFlowTracking(frame,bbox)

    def track(self, frame):
        ret, track_window = self.camshift.track(frame)

        ret2, track_window2 = self.opticalflow.track(frame)

        ret2 = ((track_window2[0],track_window2[1]),(track_window2[2],track_window2[3]),ret[2])
        draw_frame = frame.copy()
        #print("ret")
        #print(ret)
        # draw observation on image
        #camshift observtion is green
        x,y,w,h = track_window;
        draw_frame = cv.rectangle(draw_frame, (x,y), (x+w,y+h), (0,255,0),2);
        #opticalflow observtion is red
        x,y,w,h = track_window2;
        draw_frame = cv.rectangle(draw_frame, (x,y), (x+w,y+h), (0,0,255),2);

        # extract centre of camshift observation as points
        pts = cv.boxPoints(ret)

        pts = np.int0(pts)

        # use to correct kalman filter

        self.kalman.correct(imutils.center(pts));

        # extract centre of optical flow observation as points
        pts = cv.boxPoints(ret2)

        pts = np.int0(pts)

        # use to correct kalman filter

        self.kalman.correct(imutils.center(pts));

        # get new kalman filter prediction

        prediction = self.kalman.predict();

        #print("predictions")
        #print(prediction)

        x = prediction[0]-(0.5*w)

        y = prediction[1]-(0.5*h)

        w = prediction[0]+(0.5*w)-x

        h = prediction[1]+(0.5*h)-y


        prediction_box = x,y,w,h

        #print(prediction_box)

        #draw prediction in image
        #frame = cv.rectangle(frame, (prediction[0]-(0.5*w),prediction[1]-(0.5*h)), (prediction[0]+(0.5*w),prediction[1]+(0.5*h)), (0,255,0),2);

        return ret, prediction_box


class ParticleFilterTrackingCshift:
    def __init__(self, w, h, nparticles, frame, bbox):
        self.calculated = []
        self.initialise_defaults(w,h, nparticles)
        self.camshift = CAMShiftTracking(frame,bbox)

    def initialise_defaults(self, w, h, nparticles):
        """ This routine fills in the data structures with default constant
        values. It could be enhanced by reading informatino from the
        command line to allow e.g. """

        dim = 2
        nParticles = nparticles
        self.xRange = w
        self.yRange = h
        LB = [0.0, 0.0]
        UB = [self.xRange, self.yRange]
        self.model = cons.Condensation(dim, dim, nParticles)
        self.model.cvConDensInitSampleSet(LB, UB)
        self.model.DynamMatr = [[1.0, 0.0],
                     [0.0, 1.0]]

    def restart(self, frame, bbox):
        self.camshift = CAMShiftTracking(frame,bbox)

    def track(self, frame, draw=True):
        ret, track_window = self.camshift.track(frame)
        x,y,w,h = track_window;

        #pts = imutils.center(track_window)
        center_x = int(x + w/2.0)
        center_y = int(y + h/2.0)
        pts = np.array([center_x,center_y])
        for z in range(self.model.SamplesNum):

            #Calculate the confidence based on the observations
            diffX = (pts[0] - self.model.flSamples[z][0])/self.xRange
            diffY = (pts[1] - self.model.flSamples[z][1])/self.yRange
            self.model.flConfidence[z] = 1.0/(np.sqrt(np.power(diffX,2) + \
                                       np.power(diffY,2)))

        # Updates
        self.model.cvConDensUpdateByTime()
        meanPos = self.update_after_iterating(frame,x,y,draw)
        x2 = int(meanPos[0] - w/2)
        y2 = int(meanPos[1] - h/2)
        tracker_box = x2, y2, w, h
        return 1, tracker_box

    def drawCross(self, img, center, color, d):
        #On error change cv2.CV_AA for cv2.LINE_AA
        # (for differents versions of OpenCV)
        cv.line(img, (center[0] - d, center[1] - d), \
                 (center[0] + d, center[1] + d), color, 2, cv.LINE_AA, 0)
        cv.line(img, (center[0] + d, center[1] - d), \
                 (center[0]- d, center[1] + d), color, 2, cv.LINE_AA, 0)

    def update_after_iterating(self, img, x, y, draw=True):

        mean = self.model.State
        meanInt = [int(s) for s in mean]

        for j in range(len(self.model.flSamples)):
            posNew = [int(s) for s in self.model.flSamples[j]]

        if draw == True:
            for j in range(len(self.model.flSamples)):
                posNew = [int(s) for s in self.model.flSamples[j]]
                self.drawCross(img, posNew, (255, 255, 0), 2)
        else:
            for j in range(len(self.model.flSamples)):
                posNew = [int(s) for s in self.model.flSamples[j]]

        self.calculated.append(meanInt)

        for z in range(len(self.calculated)-1):
            p1 = (self.calculated[z][0], self.calculated[z][1])
            p2 = (self.calculated[z+1][0], self.calculated[z+1][1])
            #cv2.line(img, p1, p2, (255,255,255), 1)

        #print ("Mean: ", (meanInt[0], meanInt[1]))
        #print ("Real: ", (x, y))
        #print ('+++++++++++++++')
        if draw == True:
            self.drawCross(img, meanInt, (255, 0, 255), 2)
        return meanInt

#global method to instatiate the object of a selected class
def selectTracker(tracker_type, frame, bbox, frame_w, frame_h):
    nparticles = 100
    if tracker_type == 'boosting':
        tracker = boostingTracking(frame,bbox)
    elif tracker_type == 'mil':
        tracker = milTracking(frame,bbox)
    elif tracker_type == 'kcf':
        tracker = kcfTracking(frame,bbox)
    elif tracker_type == 'tld':
        tracker = tldTracking(frame,bbox)
    elif tracker_type == 'medianFlow':
        tracker = medianFlowTracking(frame,bbox)
    elif tracker_type == 'goturn':
        tracker = goturnTracking(frame,bbox)
    elif tracker_type == 'meanshift':
        tracker = MeanShiftTracking(frame,bbox)
    elif tracker_type == 'camshift':
        tracker = CAMShiftTracking(frame,bbox)
    elif tracker_type == 'optical_flow':
        tracker = OpticalFlowTracking(frame,bbox)
    elif tracker_type == 'optical_flow_orb':
        tracker = OpticalFlowTrackingORB(frame,bbox)
    elif tracker_type == 'kalman_camshift':
        tracker = KalmanTrackingCshift(frame,bbox)
    elif tracker_type == 'kalman_camshift_opticalflow':
        tracker = KalmanTrackingCshiftOflow(frame,bbox)
    elif tracker_type == 'particle_filter_camshift':
        tracker = ParticleFilterTrackingCshift(frame_w, frame_h, nparticles, frame, bbox)

    return tracker
