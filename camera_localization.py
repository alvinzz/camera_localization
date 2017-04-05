
# coding: utf-8

# In[19]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.path as path
from scipy.misc import imread, imshow, imresize
from scipy.stats import mode
from camera_simulation import get_view, world_to_im_coordinates
from plot_cameras import plot_cameras
import os

# In[86]:
PATTERN_LEN = 8.8
CAMERA_RESOLUTION = (2448, 3264)
FOCAL_LEN = 2822 # based on a 4.15mm focal length, a 3264x2448 picture resolution, and a 4.8x3.6mm sensor size
PATTERN_INTERNAL_LEN = PATTERN_LEN * 250/330

class PatternFinder:
    # Read in the pattern and image.
    def __init__(self, pattern_name, im_name, xstride=10, ystride=10):
        self.xstride = xstride
        self.ystride = ystride
        self.im_name = im_name
        self.pattern = imread(pattern_name, flatten=True)
        im = imread(im_name, flatten=True)
        im = imresize(im, (im.shape[0] // xstride, im.shape[1] // ystride))
        tempim = np.empty(im.shape[::-1], dtype=np.uint8)
        for y in range(-tempim.shape[0] // 2, (tempim.shape[0] + 1) // 2):
            for x in range(-tempim.shape[1] // 2, (tempim.shape[1] + 1) // 2):
                tempim[y + tempim.shape[0] // 2][x + tempim.shape[1] // 2] \
                    = im[-x + tempim.shape[1] // 2 - 1][y + tempim.shape[0] // 2 - 1]
        im = tempim
        _, self.im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)

    # Find the coordinates of the corners of the QR code in an image.
    def find_pattern(self):
        _, contours, hierarchy = cv2.findContours(self.im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Find the contour with the most child contours. This should be the white sheet of paper.
        paper_contour = contours[mode(hierarchy[:,:,3][0])[0][0]]
        temp_paper_contour = []
        for p in paper_contour:
            temp_paper_contour.append(p[0])
        paper_contour = np.array(temp_paper_contour)
        # Make a padded border for the sheet of paper.
        center = np.mean(paper_contour, axis=0)
        for p in paper_contour:
            if p[0] + 5 < center[0]:
                p[0] += 5
            if p[0] - 5 > center[0]:
                p[0] -= 5
            if p[1] + 5 < center[1]:
                p[1] += 5
            if p[1] - 5 > center[1]:
                p[1] -= 5
        paper_border = path.Path(paper_contour)
        # Get the extrema of the black points within the borders of the paper.
        black_coords = np.vstack(np.where(self.im.T == 0)).T
        in_paper = paper_border.contains_points(black_coords)
        black_coords_in_paper = np.array([coord for index, coord in enumerate(black_coords) if in_paper[index]])
        self.xmin = min(black_coords_in_paper, key=lambda k: k[0])
        self.ymin = min(black_coords_in_paper, key=lambda k: k[1])
        self.xmax = max(black_coords_in_paper, key=lambda k: k[0])
        self.ymax = max(black_coords_in_paper, key=lambda k: k[1])
        return self.xmin, self.ymin, self.xmax, self.ymax

    def estimate_camera_pose(self, precision=4.):
        print("Estimating camera pose for " + self.im_name + "...")
        # Uses random walk with decreasing step size. Give up if we've been
        # stuck with the same guess for a long time.
        # Attempted gradient descent, both by hand and with numpy's fsolve,
        # but got stuck in local minima both ways.
        guess = [0, 0, 80, 90, 0, 0]
        loss = precision + 1
        step = 0
        stuck_time = 0
        while loss > precision:
            if stuck_time > 5000:
                break
            stuck_time += 1
            step += 1
            for index in range(6):
                old_loss = self._evaluate_pose_guess(guess)
                # Propose a direction to move in.
                new_guess = np.array(list(guess), dtype=np.float)
                incr = max(.75, min(2000/step, 20)) # Tune-able hyperparams here.
                new_guess += incr * np.random.uniform(-1, 1, 6)
                # Limits on reasonable values.
                if new_guess[2] < 20:
                    new_guess[2] = 20
                if new_guess[3] < 10:
                    new_guess[3] = 10
                if new_guess[3] > 170:
                    new_guess[3] = 170
                if new_guess[4] < -180:
                    new_guess[4] = -180
                if new_guess[4] > 180:
                    new_guess[4] = 180
                if new_guess[5] < -180:
                    new_guess[5] = -180
                if new_guess[5] > 180:
                    new_guess[5] = 180
                # Accept the step if it's an improvement.
                new_loss = self._evaluate_pose_guess(new_guess)
                if new_loss < old_loss:
                    guess = new_guess
                    stuck_time = 0
                loss = min(new_loss, old_loss)
        # Rotate the camera at 90 deg intervals about the z-axis to see which
        # matches best with the rest of the image.
        def _rotate_90_ccw(guess):
            return [-guess[1], guess[0], guess[2], guess[3], guess[4] + 90, guess[5]]
        guesses = [guess]
        for _ in range(3):
            guesses.append(_rotate_90_ccw(guesses[-1]))
        scores = []
        for guess in guesses:
            xc, yc, zc, pitch, yaw, roll = guess
            scores.append(np.sum(self.im == get_view(xc, yc, zc, pitch, yaw, roll, self.xstride, self.ystride)))
        guess = guesses[np.argmax(scores)]
        # Display the final result!
        self._evaluate_pose_guess(guess, show_proj=True)
        print("Estimated (x, y, z, pitch, yaw, roll) for " + self.im_name + ":\n" \
            + "    ({0}, {1}, {2}, {3}, {4}, {5})" \
            .format(int(xc), int(yc), int(zc), int(pitch), int(yaw), int(roll)))
        return guess

    def _evaluate_pose_guess(self, guess, show_proj=False):
        xc, yc, zc, pitch, yaw, roll = guess
        xp1, yp1 = world_to_im_coordinates(-PATTERN_INTERNAL_LEN / 2, PATTERN_INTERNAL_LEN / 2, 0, xc, yc, zc, pitch, yaw, roll, self.xstride, self.ystride)
        xp2, yp2 = world_to_im_coordinates(-PATTERN_INTERNAL_LEN / 2, -PATTERN_INTERNAL_LEN / 2, 0, xc, yc, zc, pitch, yaw, roll, self.xstride, self.ystride)
        xp3, yp3 = world_to_im_coordinates(PATTERN_INTERNAL_LEN / 2, -PATTERN_INTERNAL_LEN / 2, 0, xc, yc, zc, pitch, yaw, roll, self.xstride, self.ystride)
        xp4, yp4 = world_to_im_coordinates(PATTERN_INTERNAL_LEN / 2, PATTERN_INTERNAL_LEN / 2, 0, xc, yc, zc, pitch, yaw, roll, self.xstride, self.ystride)
        xp1, xp2, xp3, xp4 = xp1 + CAMERA_RESOLUTION[0] // 2 // self.xstride, \
            xp2 + CAMERA_RESOLUTION[0] // 2 // self.xstride, \
            xp3 + CAMERA_RESOLUTION[0] // 2 // self.xstride, \
            xp4 + CAMERA_RESOLUTION[0] // 2 // self.xstride
        yp1, yp2, yp3, yp4 = yp1 + CAMERA_RESOLUTION[1] // 2 // self.ystride, \
            yp2 + CAMERA_RESOLUTION[1] // 2 // self.ystride, \
            yp3 + CAMERA_RESOLUTION[1] // 2 // self.ystride, \
            yp4 + CAMERA_RESOLUTION[1] // 2 // self.ystride
        if show_proj:
            plt.figure(1)
            plt.subplot(121)
            plt.title("Detected QR corners for " + self.im_name)
            plt.imshow(self.im, cmap='gray')
            plt.plot(self.xmin[0], self.xmin[1], marker='o', color='g')
            plt.plot(self.ymin[0], self.ymin[1], marker='o', color='g')
            plt.plot(self.xmax[0], self.xmax[1], marker='o', color='g')
            plt.plot(self.ymax[0], self.ymax[1], marker='o', color='g')

            plt.subplot(122)
            plt.title("Projected view for camera w/ pose: \n \
                (x, y, z, pitch, ya, roll) = ({0}, {1}, {2}, {3}, {4}, {5})" \
                .format(int(xc), int(yc), int(zc), int(pitch), int(yaw), int(roll)))
            projection = get_view(xc, yc, zc, pitch, yaw, roll, self.xstride, self.ystride)
            plt.imshow(projection, cmap='gray')
            plt.plot(self.xmin[0], self.xmin[1], marker='o', color='g')
            plt.plot(self.ymin[0], self.ymin[1], marker='o', color='g')
            plt.plot(self.xmax[0], self.xmax[1], marker='o', color='g')
            plt.plot(self.ymax[0], self.ymax[1], marker='o', color='g')
            plt.plot(xp1, yp1, marker='o', color='b')
            plt.plot(xp2, yp2, marker='o', color='b')
            plt.plot(xp3, yp3, marker='o', color='b')
            plt.plot(xp4, yp4, marker='o', color='b')
            plt.show()
        loss = np.sqrt((xp1 - self.xmin[0])**2 + (yp1 - self.xmin[1])**2 \
            + (xp2 - self.ymin[0])**2 + (yp2 - self.ymin[1])**2 \
            + (xp3 - self.xmax[0])**2 + (yp3 - self.xmax[1])**2 \
            + (xp4 - self.ymax[0])**2 + (yp4 - self.ymax[1])**2)
        return loss

if __name__ == "__main__":
    IMDIR = './'
    img_names = []
    for file in os.listdir(IMDIR):
        if file.endswith(".JPG"):
            img_names.append(file)

    poses = []
    for img_name in img_names:
        pf = PatternFinder('pattern.png', img_name)
        pf.find_pattern()
        poses.append(pf.estimate_camera_pose())

    positions = [pose[:3] for pose in poses]
    orientations = [pose[3:] for pose in poses]
    cam_ids = [name[4:-4] for name in img_names]
    plot_cameras(positions, orientations, cam_ids, animated=False)
