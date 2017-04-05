Camera Localization
===================

Repo for DroneDeploy's camera localization challenge.

Run camera_localization.py to get camera pose estimates for each image in the root folder.

## Method

First, load in and downsample the image by a factor of 10, to save time.
Also, load the pattern and threshold the image.

Locate the QR code within the image.
* Use cv2.findContours to find the boundaries of the piece of paper within the image.
* Identify the black pixels with the minimal/maximal x/y coordinates within these borders to get the corners of the QR pattern.
![qr corner finding][camera_localization2.png]

Estimate the camera pose, using the corners of the QR pattern.
* Gradient descent and numpy's fsolve both fail to converge to a reasonable degree.
* Use a random walk with decreasing step sizes through time.
* Definitely an area for improvement.
![camera localization][camera_localization.png]

Simulate the view from four rotations about the z-axis.
* Rotating the camera 90 degrees about the z-axis will still produce the same corners.
* Run camera_simulation.py for an example of this.
* Find the best rotation by comparing the simulated image to the actual image.
![camera rotation][camera_simulation.png]

Plot the camera coordinates and their image planes!
![camera visualizer][camera_visualizer.png]
![camera visualizer2][camera_visualizer2.png]
