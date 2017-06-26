**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Normalizing the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/car_notcar_color_spaces_hog.png
[image3]: ./output_images/sliding_window.jpg
[image4]: ./output_images/unfilt_test1.jpg
[image5]: ./output_images/output.png
[image6]: ./output_images/heatmap_test1.jpg
[image7]: ./output_images/filt_test1.jpg
[video1]: ./processed_project_video.mp4

---

### Histogram of Oriented Gradients (HOG)

#### 1. HOG features extraction from the training images.

The code for this step is contained in lines 129 through 215 of the file called `search_classify.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Final choice of HOG parameters selection

I experimented with a number of different combinations of color spaces and HOG parameters and trained a linear SVM using different combinations of HOG features extracted from the YCrCb color channels. YCrCb color space with `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` provided a good result as compared to other color spaces. With these parameters I was able to achieve a classification accuracy of 98.7 % and it also helped identify the cars correctly. Hence I settled on these parameters for my project.

#### 3. Training a classifier using your selected HOG features and color features

I trained a linear SVM using all channels of images converted to YCrCb space. I included spatial features, histogram of color features and HOG features in the feature set to train my classifier. The code for this step is contained in lines 218 through 245 of the file called 'search_classify.py'. The trained classifier along with the parameters for the feature extraction were then stored in 'svc_pickle.p' file. I was able to achieve 98.7% accuracy for a feature length of 8460.

### Sliding Window Search

#### 1. Implementing a sliding window search

The sliding windoe search is implemented in lines 24 through 95 in 'hog_subsample.py' file. Here is the output of all the sliding windows drawn on the input image:

![alt text][image3]

As can be seen in the image the sliding windows extend all the way to the rightmost part of the image.

#### 2. Output of the pipeline and optimizing the performance of your classifier

On the test images the Hog sub-sampling feature extraction was done on three scales(1.3, 1.5 and 1.9 line 105 in 'hog_subsample.py'), but on the video pipeline the Hog sub-sampling was done on two scales (1.5 and 1.8). The reason for this is because sampling with 1.3 scale induced many false positives. Here is the output of multiple detections on one of the test images:

![alt text][image4]
---

### Video Implementation

#### 1. Link to your final video output

Here's the [link to my video result][video1]


#### 2. Filter implementation for false positives and combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

### Discussion

#### 1. Problems / issues faced in implementation of this project

Below are some of the improvements required for the pipeline to work efficiently:
1) In the current pipeline I am assinging the scales manually, this can be improved by having a minimum and maximum scales with intermediate scales in between them.
2) The current pipeline dosen't detect a vehicle if it is positioned at some angle as can be seen in 'filt_test5.jpg' in output_images folder. This can be improved by adding images of vehicles positioned at different angles to the training dataset.

