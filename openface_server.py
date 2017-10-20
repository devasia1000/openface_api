# Copyright (C) 2017 Devasia Manuel - All Rights Reserved

from flask import Flask
from flask import request
from openface_simple import OpenfaceSimple
import argparse

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--trainingImagesDir', type=str, default='training_images',
                    help='Directory to store training images')
args = parser.parse_args()

openface_simple = OpenfaceSimple(args.trainingImagesDir)

@app.route('/put_training_image')
def put_training_image():
    identity = request.args.get('identity')
    base_64_encoded_image = request.args.get('base_64_encoded_image')
    return openface_simple.putTrainingImage(identity, base_64_encoded_image)

@app.route('/perform_training')
def perform_training():
    return openface_simple.performTraining()

@app.route('/get_inference_result')
def get_inference_result():
    base_64_encoded_image = request.args.get('base_64_encoded_image')
    return openface_simple.getInferenceResult(base_64_encoded_image)

@app.route('/reset_training_images')
def reset_training_images():
    return openface_simple.resetTrainingImages()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)