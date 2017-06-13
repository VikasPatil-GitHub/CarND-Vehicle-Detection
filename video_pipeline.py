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
scale = 1.5

boxes = BoundingBoxes(n=30)

def process_image(img):
	out_img, box_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

	boxes.update(box_list)
	
	heat = np.zeros_like(out_img[:,:,0]).astype(np.float)

	# Add heat to each box in box list
	heat = add_heat(heat,boxes.allboxes)
		
	# Apply threshold to help remove false positives
	heat = apply_threshold(heat,10)

	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)

	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(np.copy(img), labels)
	
	return draw_img
	
out_dir='./'
output = out_dir+'processed_project_video.mp4'
clip = VideoFileClip("project_video.mp4")
out_clip = clip.fl_image(process_image) 
out_clip.write_videofile(output, audio=False)