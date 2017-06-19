import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import os
from moviepy.editor import VideoFileClip
from hog_subsample import *
from filter import *

dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

print(svc,X_scaler,orient,pix_per_cell,cell_per_block,spatial_size,hist_bins)

ystart = 370
ystop = 656
scales = [1.5,1.8]
#scales = [1.5]

boxes = BoundingBoxes(n=30)

box_list_all_scales = []

def process_image(img):
	
	image = np.copy(img)

	del box_list_all_scales[:]
	
	for scale in scales:
		box_list_all_scales.extend(find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)[0])
	
	out_img = draw_boxes(img, box_list_all_scales, color=(0, 0, 255), thick=6)

	boxes.update(box_list_all_scales)
	
	heat = np.zeros_like(out_img[:,:,0]).astype(np.float)

	# Add heat to each box in box list
	heat = add_heat(heat,boxes.allboxes)
		
	# Apply threshold to help remove false positives
	heat = apply_threshold(heat,15)

	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)

	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(image, labels)
	
	return draw_img
	
out_dir='./'
output = out_dir+'processed_project_video.mp4'
clip = VideoFileClip("project_video.mp4")
out_clip = clip.fl_image(process_image) 
out_clip.write_videofile(output, audio=False)