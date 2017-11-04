import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
from helper import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import random
import time
from scipy.ndimage.measurements import label
from sklearn.cross_validation import train_test_split
from moviepy.editor import VideoFileClip
from collections import deque

# Read in images of cars and notcars
def train_svm():
    # # Read in images of cars and notcars
    car_imgs = glob.glob('vehicles/*/*.png')
    notcar_imgs = glob.glob('non-vehicles/*/*.png')

    ### Parameters to get features and train SVC
    COLOR_SPACE = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    ORIENTATION = 9  # HOG orientations
    PIX_PER_CELL = 8  # HOG pixels per cell
    CELL_PER_BLOCK= 2  # HOG cells per block
    HOG_CHANNEL = "ALL"  # Can be 0, 1, 2, or "ALL"
    SPATIAL_SIZE = (32, 32)  # Spatial binning dimensions
    HIST_BINS = 32  # Number of histogram bins
    SPATIAL_FEAT = True  # Spatial features on or off
    HIST_FEAT = True  # Histogram features on or off
    HOG_FEAT = True  # HOG features on or off
    # y_start_stop = [500, None]  # Min and max in y to search in slide_window()

    car_features = extract_features(car_imgs, color_space=COLOR_SPACE,
                                    spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS,
                                    orient=ORIENTATION, pix_per_cell=PIX_PER_CELL,
                                    cell_per_block=CELL_PER_BLOCK,
                                    hog_channel=HOG_CHANNEL, spatial_feat=SPATIAL_FEAT,
                                    hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)
    notcar_features = extract_features(notcar_imgs, color_space=COLOR_SPACE,
                                       spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS,
                                       orient=ORIENTATION, pix_per_cell=PIX_PER_CELL,
                                       cell_per_block=CELL_PER_BLOCK,
                                       hog_channel=HOG_CHANNEL, spatial_feat=SPATIAL_FEAT,
                                       hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', ORIENTATION, 'orientations,', PIX_PER_CELL,
          'pixels per cell, and', CELL_PER_BLOCK, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC(C=1.5)
    svc.fit(X_train, y_train)
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    classifier_data = {}
    classifier_data["svc"] = svc
    classifier_data["scaler"] = X_scaler
    classifier_data["orient"] = ORIENTATION
    classifier_data["pix_per_cell"] = PIX_PER_CELL
    classifier_data["cell_per_block"] = CELL_PER_BLOCK
    classifier_data["spatial_size"] = SPATIAL_SIZE
    classifier_data["hist_bins"] = HIST_BINS

    with open('svc_data.pickle', 'wb') as handle:
        pickle.dump(classifier_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def process_test_images(test_images):



    def process_image(img):
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        boxes = []
        ystart = 380
        ystop = 720
        xstart = 600
        xstop = 1280
        scales = [1.0, 1.5, 2.0]
        orient = 9
        pix_per_cell = 8  # HOG pixels per cell
        cell_per_block = 2  # HOG cells per block
        spatial_size = (32, 32)  # Spatial binning dimensions
        hist_bins = 32  # Number of histogram bins

        for scale in scales:
            out_img, box_list = find_cars(image, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient,
                                          pix_per_cell,
                                          cell_per_block, spatial_size, hist_bins)
            boxes.extend(box_list)

        # Add heat to each box in box list
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        heat = add_heat(heat, boxes)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)

        return (heatmap, draw_img)

    # testing it all test images
    # test_images = glob.glob('./CarND-Vehicle-Detection/test_images/*.jpg')
    fig = plt.figure(figsize=(15, 10))
    n_imgs = len(test_images)

    for i in range(n_imgs):
        image = cv2.imread(test_images[i])
        heatmap,draw_img= process_image(image)
        # visualize outputs on all test images
        plt.subplot(n_imgs, 2, 2 * (i + 1) - 1)
        plt.imshow(draw_img)
        plt.axis('off')
        plt.subplot(n_imgs, 2, 2 * (i + 1))
        plt.imshow(heatmap, cmap='hot')
        plt.axis('off')
    fig.savefig('./writeup_imgs/heat_map.png')
    plt.show()

def process_video(video_path):
    history = deque(maxlen=8)

    def process_frame(img):
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        boxes = []
        ystart = 380
        ystop = 720
        xstart = 600
        xstop = 1280
        scales = [1.0, 1.5, 2.0]
        orient = 9
        pix_per_cell = 8  # HOG pixels per cell
        cell_per_block = 2  # HOG cells per block
        spatial_size = (32, 32)  # Spatial binning dimensions
        hist_bins = 32  # Number of histogram bins

        for scale in scales:
            out_img, box_list = find_cars(image, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient,
                                          pix_per_cell,
                                          cell_per_block, spatial_size, hist_bins)
            boxes.extend(box_list)

        # Add heat to each box in box list
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        heat = add_heat(heat, boxes)

        history.append(heat)
        mean_heat = np.mean(history,axis=0)
        # fig = plt.figure(figsize=(10., 5))
        # plt.imshow(mean_heat)


        # Apply threshold to help remove false positives
        heat = apply_threshold(mean_heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)

        return draw_img

    clip = VideoFileClip(video_path)
    output_clip = clip.fl_image(process_frame)
    output_clip.write_videofile('output_c1_5.mp4',audio=False)

if __name__ == '__main__':
    ## train svm, reload paras
    # train_svm()
    dist_pickle = pickle.load(open("svc_data.pickle", "rb"))
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]

## test images
    test_images = glob.glob('./test_images/*.jpg')
    # process_test_images(test_images)

  ## video
    # video_path =  "./test_video.mp4"#"./project_video.mp4"
    video_path = "./project_video.mp4"
    process_video(video_path)

