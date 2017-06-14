import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import os
from lesson_functions import *
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


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
	box_list=[]
	all_boxes = []
	lengths = []
	draw_img = np.copy(img)
	img = img.astype(np.float32)/255

	img_tosearch = img[ystart:ystop,:,:]
	ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
	if scale != 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
		
	ch1 = ctrans_tosearch[:,:,0]
	ch2 = ctrans_tosearch[:,:,1]
	ch3 = ctrans_tosearch[:,:,2]

	# Define blocks and steps as above
	nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
	nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
	nfeat_per_block = orient*cell_per_block**2

	# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
	window = 64
	nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	cells_per_step = 2  # Instead of overlap, define how many cells to step
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step

	# Compute individual channel HOG features for the entire image
	hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step
			# Extract HOG for this patch
			hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell

			_xbox_left = np.int(xleft*scale)
			_ytop_draw = np.int(ytop*scale)
			_win_draw = np.int(window*scale)
			all_boxes.append(((_xbox_left, _ytop_draw+ystart), (_xbox_left+_win_draw,_ytop_draw+_win_draw+ystart)))
			
			# Extract the image patch
			subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
			# Get color features
			spatial_features = bin_spatial(subimg, size=spatial_size)
			hist_features = color_hist(subimg, nbins=hist_bins)

			# Scale features and make a prediction
			test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
			#test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
			test_prediction = svc.predict(test_features)
			
			if test_prediction == 1:
				xbox_left = np.int(xleft*scale)
				ytop_draw = np.int(ytop*scale)
				win_draw = np.int(window*scale)
				box_list.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)))
				cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
		lengths.append(len(all_boxes))
	return draw_img, box_list
    
ystart = 370
ystop = 656
scale = 1.65

image_test = mpimg.imread('./test_images/test4.jpg')

out_img, box_list = find_cars(image_test, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

window_img = draw_boxes(image_test, box_list, color=(0, 0, 255), thick=6)


plt.imshow(window_img)
plt.title('Sliding window output')
plt.show()

font_size=15
f, axarr = plt.subplots(6, 3)
f.subplots_adjust(hspace=0.3)

images_test = glob.glob('./test_images/test*.jpg')
ind = 0
for image_test in images_test:
	img = mpimg.imread(image_test)
	out_img, box_list = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

	heat = np.zeros_like(out_img[:,:,0]).astype(np.float)

	# Add heat to each box in box list
	heat = add_heat(heat,box_list)
		
	# Apply threshold to help remove false positives
	heat = apply_threshold(heat,1)

	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)

	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(np.copy(img), labels)

	plt.imsave('./output_images/unfilt_'+os.path.split(image_test)[1],out_img)
	
	axarr[ind,0].imshow(out_img)
	axarr[ind,0].set_xticks([])
	axarr[ind,0].set_yticks([])
	title = "Test image {0}".format(ind)
	axarr[ind,0].set_title(title, fontsize=font_size)

	plt.imsave('./output_images/heatmap_'+os.path.split(image_test)[1],heat)
	
	axarr[ind,1].imshow(heat,cmap='hot')
	axarr[ind,1].set_xticks([])
	axarr[ind,1].set_yticks([])
	title = "Test image {0} heatmap".format(ind)
	axarr[ind,1].set_title(title, fontsize=font_size)
	
	plt.imsave('./output_images/filt_'+os.path.split(image_test)[1],draw_img)
	
	axarr[ind,2].imshow(draw_img)
	axarr[ind,2].set_xticks([])
	axarr[ind,2].set_yticks([])
	title = "Test image {0} output".format(ind)
	axarr[ind,2].set_title(title, fontsize=font_size)
	
	ind += 1

plt.show()