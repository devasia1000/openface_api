# Copyright (C) 2017 Devasia Manuel - All Rights Reserved

import argparse
import os
import base64
import numpy as np
from PIL import Image
import hashlib
import StringIO
import openface

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

modelDir = os.path.join(os.getcwd(), 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')
args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
    cuda=args.cuda)

class OpenfaceSimple:

    #################################################################
    ############### PRIVATE MEMBER FUNCTIONS ########################
    #################################################################

    def __init__(self, training_dir):
        self.training_dir = training_dir
        self.base64_head = 'data:image/jpeg;base64,'

    def updateIdentityMaps(self):
        self.identity_map = self.getIdentityMap()
        self.reverse_identity_map = {v: k for k, v in self.identity_map.iteritems()}

    def getFaceVector(self, img):
        buf = np.fliplr(np.asarray(img))
        rgbFrame = np.zeros((img.size[1], img.size[0], 3), dtype=np.uint8)
        rgbFrame[:, :, 0] = buf[:, :, 2]
        rgbFrame[:, :, 1] = buf[:, :, 1]
        rgbFrame[:, :, 2] = buf[:, :, 0]

        bb = align.getLargestFaceBoundingBox(rgbFrame)
        if bb is None:
            return ('ERROR: could not find bounding box of largest face', None)
        landmarks = align.findLandmarks(rgbFrame, bb)
        if landmarks is None:
            return ('ERROR: could not find landmarks of face', None)
        alignedFace = align.align(args.imgDim, rgbFrame, bb,
            landmarks=landmarks, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            return ('ERROR: could not align face', None)

        rep = net.forward(alignedFace)
        if rep is None:
            return ('ERROR: could not get face embedding from neural network', None)
        return ('Succesfully retrieved face embedding', rep)

    def getIdentityMap(self):
        idens = []
        for root, dirs, files in os.walk(self.training_dir, topdown=False):
            for filename in files:
                path = os.path.join(root, filename)
                iden = path.split('/')[-2]
                if iden not in idens:
                    idens.append(iden)
        idens = sorted(idens)
        return {k: v for v, k in enumerate(idens)}

    def getTrainingData(self):
        X = []
        y = []

        for root, dirs, files in os.walk(self.training_dir, topdown=False):
            for filename in files:
                path = os.path.join(root, filename)
                iden = path.split('/')[-2]
                message, rep = self.getFaceVector(Image.open(path))
                if rep is None:
                    continue
                X.append(rep)
                y.append(self.identity_map[iden])

        X = np.vstack(X)
        y = np.array(y)
        return (X, y)

    def trainSVM(self):
        X, y = self.getTrainingData()

        if len(X) < 10 or len(y) < 2:
            print 'ERROR: need at least 5 images person for at least 2 different people to train model'
            return

        param_grid = [
                {'C': [1, 10, 100, 1000],
                 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf']}
            ]
        self.svm = GridSearchCV(SVC(C=1), param_grid, cv=5).fit(X, y)

    def getSVMInference(self, img):
        tempBuff = StringIO.StringIO()
        tempBuff.write(img)
        tempBuff.seek(0)
        img_arr = Image.open(tempBuff)

        message, rep = self.getFaceVector(img_arr)
        rep = rep.reshape(1, -1) # reshape since it contains a single sample
        iden = self.svm.predict(rep)[0]
        return self.reverse_identity_map[iden]

    def isBase64EncodedImage(self, img):
        if not img and not img.startswith(self.base64_head):
            return False
        return True

    #################################################################
    ############### PUBLIC MEMBER FUNCTIONS #########################
    #################################################################

    def putTrainingImage(self, identity, base_64_encoded_image):
        if not identity:
            return 'ERROR: identity is empty'
        if not self.isBase64EncodedImage(base_64_encoded_image):
            return 'ERROR: Is not a valid base 64 encoded JPEG image'

        training_dir = os.path.join(self.training_dir, identity)
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)

        filename = os.path.join(training_dir,
            hashlib.md5(base_64_encoded_image).hexdigest()[:6]) + '.jpeg'

        fh = open(filename, 'wb')
        imgdata = base64.b64decode(base_64_encoded_image[len(self.base64_head):])
        fh.write(imgdata)
        fh.close()

        return 'Success'

    def performTraining(self):
        self.updateIdentityMaps()
        self.trainSVM()
        return 'Success'

    def getInferenceResult(self, base_64_encoded_image):
        if not self.isBase64EncodedImage(base_64_encoded_image):
            return 'ERROR: Is not a valid base 64 encoded JPEG image'
        imgdata = base64.b64decode(base_64_encoded_image[len(self.base64_head):])
        return self.getSVMInference(imgdata)

    def resetTrainingImages(self):
        for root, dirs, files in os.walk(self.training_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        self.svm = None
        return 'Success'
