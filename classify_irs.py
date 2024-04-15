from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
from PIL import Image
import numpy as np
import argparse
import cv2
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import cv2
import os
# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
import argparse

#two function definitions up here

def quantify_image(image):
	''' compute the histogram of oriented gradients feature vector for
	 the input image'''
	features = feature.hog(image, orientations=9, 
						#gradients will be computed for 9 orientation angles
		pixels_per_cell=(10, 10), cells_per_block=(2, 2),
		transform_sqrt=True, block_norm="L1")
	
	'''transform_sqrt=True: This parameter indicates whether 
	the square root of each cell's histogram values 
	should be computed before passing them to the block normalization stage. 
	This is often done to improve the invariance to changes in illumination.
	 return the feature vector'''

	'''block_norm="L1": This parameter specifies the type of block normalization to be applied. 
	Here, "L1" normalization is used, which means each block's histogram values 
	will be normalized using L1 norm.'''


	return features


def load_split(path):

	''' grab the list of images in the input directory, then initialize
	 the list of data (i.e., images) and class labels'''
	
	imagePaths = list(paths.list_images(path))
	data = []
	labels = []
	# loop over the image paths
	for imagePath in imagePaths:
		# extract the class label from the filename
		label = imagePath.split(os.path.sep)[-2]

		'''This line splits the imagePath string using the operating system's path separator 
		(os.path.sep). This separator (/ in Unix-based systems, \ in Windows) separates the 
		directories and file name in a file path. After splitting, the [-2] index indicates 
		that the code selects the second-to-last element of the resulting list. 
		This is because the last element typically represents the file name itself, 
		and the one before it would be the immediate parent directory. 
		This extracted segment is assigned to the variable label, 
		presumably to represent the class label associated with the image.'''

		''' load the input image, convert it to grayscale, and resize
		 it to 200x200 pixels, ignoring aspect ratio'''
		
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #RGB to grayscale
		image = cv2.resize(image, (200, 200))

		''' threshold the image such that the drawing appears as white
		 on a black background
		this uses Otsu thresholding
		 2nd arg = 0 indicates otsu
		 3rd arg = 255 indicates that maximum value that will be used for the 
		thresholding operation. 
		Pixels with values above the threshold will be set to this value.
		cv2.THRESH_BINARY_INV specifies that the pixels below the 
		threshold will be set to the maximum value (255) 
		and pixels above the threshold will be set to 0. 
		cv2.THRESH_OTSU indicates that the threshold value 
		will be determined using Otsu's method.'''

		image = cv2.threshold(image, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		
		# quantify the image
		features = quantify_image(image)
		# update the data and labels lists, respectively
		data.append(features)
		labels.append(label)
	# return the data and labels
	return (np.array(data), np.array(labels))

# construct the argument parser and parse the arguments
'''ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="knn",
	help="type of python machine learning model to use")
args = vars(ap.parse_args())'''
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-t", "--trials", type=int, default=5,
	help="# of trials to run")
args = vars(ap.parse_args())
'''My script handles two command line arguments:

--dataset : The path to the input dataset (either waves or spirals).
--trials : The number of trials to run (by default we run 5 trials).
 define the path to the training and testing directories'''

trainingPath = os.path.sep.join([args["dataset"], "training"])

''' This line constructs the path to the training directory by joining the 
value of args["dataset"] (which presumably 
contains the root directory of the dataset) with the subdirectory "training".
example below'''
'''components = ['folder', 'subfolder', 'file.txt']
path = os.path.sep.join(components)'''

#result will be folder\subfolder\file.txt  # On Windows



testingPath = os.path.sep.join([args["dataset"], "testing"])
# loading the training and testing data
print("[INFO] loading data...")
(trainX, trainY) = load_split(trainingPath)
(testX, testY) = load_split(testingPath)
# encode the labels as integers
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)
# initialize our trials dictionary
trials = {}

# loop over the number of trials to run
for i in range(0, args["trials"]):
	# train the model
	print("[INFO] training model {} of {}...".format(i + 1,
		args["trials"]))
	model = RandomForestClassifier(n_estimators=100)
	#n_estimators = number of decision trees
 

	model.fit(trainX, trainY)
	'''This line trains the random forest classifier model on the training data. 
 The fit() method fits the model to the training data (trainX, trainY), 
where trainX contains the input features and trainY contains the corresponding labels.'''
	
	
	''' make predictions on the testing data and initialize a dictionary
	 to store our computed metrics'''
	
	predictions = model.predict(testX)
	metrics = {}

	'''compute the confusion matrix and and use it to derive the raw
	 accuracy, sensitivity, and specificity'''
	
	cm = confusion_matrix(testY, predictions).flatten()
	(tn, fp, fn, tp) = cm
	metrics["acc"] = (tp + tn) / float(cm.sum())
	metrics["sensitivity"] = tp / float(tp + fn)
	metrics["specificity"] = tn / float(tn + fp)
	
	'''These lines calculate and store performance metrics based on the values 
 derived from the confusion matrix. 
 Specifically, it computes accuracy (acc), sensitivity (true positive rate), 
 and specificity (true negative rate).
	 loop over the metrics'''
	
	for (k, v) in metrics.items():
		''' update the trials dictionary with the list of values for
		 the current metric'''
		l = trials.get(k, [])
		l.append(v)
		trials[k] = l


	'''These lines update the trials dictionary with the computed performance metrics for the current trial. 
	It retrieves the list of values associated with the current metric k from the trials dictionary (or creates an empty list if the metric does not exist yet), 
	appends the new value v to the list, 
	and then updates the trials dictionary with the updated list of values for the current metric.'''

# loop over our metrics
for metric in ("acc", "sensitivity", "specificity"):
	# grab the list of values for the current metric, then compute
	# the mean and standard deviation
	values = trials[metric]
	mean = np.mean(values)
	std = np.std(values)
	# show the computed metrics for the statistic
	print(metric)
	print("=" * len(metric))
	#This line prints a line of equals signs (=) that is the same length as the metric name printed in the previous line.
	
	print("u={:.4f}, o={:.4f}".format(mean, std))
	#The "{:.4f}" format specifier formats the floating-point numbers to have four decimal places.
	
	print("")

# randomly select a few images and then initialize the output images
# for the montage
testingPaths = list(paths.list_images(testingPath))
'''This line retrieves a list of file paths to images in the testing directory
 using the list_images() function from the paths module. testingPath likely 
 contains the path to the testing directory. list_images() returns a generator 
 of file paths, which is converted into a list using list() and assigned to the
   variable testingPaths.'''




idxs = np.arange(0, len(testingPaths))
'''This line creates an array of indices from 0 to the length of testingPaths 
(exclusive). It uses NumPy's arange() function to generate a range of integers 
and assigns the result to the variable idxs.'''




idxs = np.random.choice(idxs, size=(25,), replace=False)
images = []
# loop over the testing samples
for i in idxs:
	# load the testing image, clone it, and resize it
	image = cv2.imread(testingPaths[i])
	output = image.copy()
	output = cv2.resize(output, (128, 128))
	'''Inside the loop, this block of code loads an image from the testingPaths 
	list at the index i using OpenCV's imread() function. It then creates a 
	copy of the image using copy() method and resizes the copy to a fixed size
	  of (128, 128) pixels using OpenCV's resize() function. 
	  The resized image is stored in the output variable.'''
	

	# pre-process the image in the same manner we did earlier
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (200, 200))
	image = cv2.threshold(image, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	'''This block of code preprocesses the original image (image) before 
	resizing it. It converts the image to grayscale using cvtColor() function, 
	resizes it to (200, 200) pixels, and applies thresholding using threshold() 
	function with the parameters THRESH_BINARY_INV | THRESH_OTSU.
	 This preprocessing step likely converts the image into a binary format 
	 suitable for further analysis or processing.
	 cv2.THRESH_BINARY_INV specifies that the pixels below the 
		threshold will be set to the maximum value (255) 
		and pixels above the threshold will be set to 0. 
		cv2.THRESH_OTSU indicates that the threshold value 
		will be determined using Otsu's method.'''
	
	# quantify the image and make predictions based on the extracted
	# features using the last trained Random Forest
 
 #quantify_images returns a bunch of features
	features = quantify_image(image)
	preds = model.predict([features])
	'''This line uses the trained model (presumably a machine learning model) to 
	make predictions on the features extracted from the image. 
	The predict() method takes the features as input and returns the 
	predicted class label(s).'''

	label = le.inverse_transform(preds)[0]
	# draw the colored class label on the output image and add it to
	# the set of output images
	color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
	# If the predicted label is "healthy", the color is set to green (0, 255, 0); otherwise, it is set to red (0, 0, 255).
	
	cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
		color, 2)
	images.append(output)
# create a montage using 128x128 "tiles" with 5 rows and 5 columns
montage = build_montages(images, (128, 128), (5, 5))[0]
'''This line creates a montage (a grid-like arrangement) of the output images 
(images) using the build_montages() function. The montage is created with 5 rows 
and 5 columns of tiles, each of size 128x128 pixels. 
The [0] indexing is used to extract the montage from the list returned by 
build_montages().'''
# show the output montage
cv2.imshow("Output", montage)
cv2.waitKey(0)

