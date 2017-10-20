# Copyright (C) 2017 Devasia Manuel - All Rights Reserved

import argparse
import glob
import base64
from openface_simple import OpenfaceSimple

head = 'data:image/jpeg;base64,'

parser = argparse.ArgumentParser()
parser.add_argument('--trainingImagesDir', type=str, default='training_images',
                    help='Directory to store training images')
args = parser.parse_args()

adrien_images = glob.glob('testing/data/adrien_*')
adrien_base64_images = []
adrien_labels = ['adrien'] * len(adrien_images)

ann_images = glob.glob('testing/data/ann_*')
ann_base64_images = []
ann_labels = ['ann'] * len(ann_images)

anna_images = glob.glob('testing/data/anna_*')
anna_base64_images = []
anna_labels = ['anna'] * len(anna_images)

def setup():
	# Read images of 'adrien' from testing/data and convert them into Base64 encoded JPEGs
	for img_path in adrien_images:
		with open(img_path, 'rb') as image_file:
			encoded_string = head + base64.b64encode(image_file.read())
		adrien_base64_images.append(encoded_string)

	# Read images of 'anna' from testing/data and convert them into Base64 encoded JPEGs
	for img_path in anna_images:
		with open(img_path, 'rb') as image_file:
			encoded_string = head + base64.b64encode(image_file.read())
		anna_base64_images.append(encoded_string)

	# Read images of 'ann' from testing/data and convert them into Base64 encoded JPEGs
	for img_path in ann_images:
		with open(img_path, 'rb') as image_file:
			encoded_string = head + base64.b64encode(image_file.read())
		ann_base64_images.append(encoded_string)

def testModel():
	# Initialize an instance of the model
	openface_simple = OpenfaceSimple(args.trainingImagesDir)

	# Add 8 training images of 'adrien' to the model while keeping 3 images aside
	# for inference. These images can be found in /testing/data/
	for (identity, img) in zip(adrien_labels[0:8], adrien_base64_images[0:8]):
		openface_simple.putTrainingImage(identity, img)

	# Add 8 training images for 'ann' to the model while keeping 3 images aside
	# for inference. These images can be found in /testing/data/
	for (identity, img) in zip(ann_labels[0:8], ann_base64_images[0:8]):
		openface_simple.putTrainingImage(identity, img)

	# Add 8 training images for 'anna' to the model while keeping 3 images aside
	# for inference. These images can be found in /testing/data/
	for (identity, img) in zip(anna_labels[0:8], anna_base64_images[0:8]):
		openface_simple.putTrainingImage(identity, img)

	# Train the model
	openface_simple.performTraining()

	# Ask the model to classify three new images from 'adrien'
	adrien_result_1 = openface_simple.getInferenceResult(adrien_base64_images[8])
	adrien_result_2 = openface_simple.getInferenceResult(adrien_base64_images[9])
	adrien_result_3 = openface_simple.getInferenceResult(adrien_base64_images[10])

	# Verify that the model correctly classified these new images as 'adrien'
	assert (adrien_result_1 == 'adrien')
	assert (adrien_result_2 == 'adrien')
	assert (adrien_result_3 == 'adrien')

	# Ask the model to classify three new images from 'ann'
	ann_result_1 = openface_simple.getInferenceResult(ann_base64_images[8])
	ann_result_2 = openface_simple.getInferenceResult(ann_base64_images[9])
	ann_result_3 = openface_simple.getInferenceResult(ann_base64_images[10])

	# Verify that the model correctly classified these new images as 'ann'
	assert (ann_result_1 == 'ann')
	assert (ann_result_2 == 'ann')
	assert (ann_result_3 == 'ann')

	# Ask the model to classify three new images from 'anna'
	anna_result_1 = openface_simple.getInferenceResult(anna_base64_images[8])
	anna_result_2 = openface_simple.getInferenceResult(anna_base64_images[9])
	anna_result_3 = openface_simple.getInferenceResult(anna_base64_images[10])

	# Verify that the model correctly classified these new images as 'anna'
	assert (anna_result_1 == 'anna')
	assert (anna_result_2 == 'anna')
	assert (anna_result_3 == 'anna')

	# Delete all training data and delete the model
	openface_simple.resetTrainingImages()

	print 'ALL TESTS PASS!'

setup()
testModel()










