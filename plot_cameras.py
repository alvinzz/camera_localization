
# coding: utf-8

# In[6]:

# Uses code from the matplotlib 3D-plotting tutorials.
from scipy.misc import imread, imshow, imresize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from camera_simulation import get_image_plane

PATTERN_LEN = 8.8
CAMERA_RESOLUTION = (2448, 3264)
FOCAL_LEN = 2822 # based on a 4.15mm focal length, a 3264x2448 picture resolution, and a 4.8x3.6mm sensor size

def plot_cameras(camera_positions, camera_orientations, camera_names, animated=False):
    camera_positions = np.array(camera_positions)
    camera_orientations = np.array(camera_orientations)

    # Initialize figure.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    # Import pattern information.
    PATTERN_SIZE = 8.8

    pattern = imread('pattern.png', flatten=True)

    # Plot pattern.
    colors = np.array([[[y/255, y/255, y/255] for y in x] for x in pattern])

    x, y = np.meshgrid(np.linspace(-PATTERN_SIZE / 2, PATTERN_SIZE / 2, pattern.shape[0]),                          np.linspace(-PATTERN_SIZE / 2, PATTERN_SIZE / 2, pattern.shape[1]))

    ax.plot_surface(x, y, np.zeros(pattern.shape), facecolors=colors, cmap='gray')

    # Plot camera markers.
    xs = camera_positions[:,0]
    ys = camera_positions[:,1]
    zs = camera_positions[:,2]

    ax.scatter(xs, ys, zs, c='b', marker='o')

    # Label cameras.
    for index, pos in enumerate(camera_positions):
        ax.text(pos[0], pos[1], pos[2], camera_names[index], color='r')

    # Label axes.
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Normalize axes so that they all have the same scale.
    xs = np.append(xs, 0)
    ys = np.append(ys, 0)
    zs = np.append(zs, 0)
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max()
    mid_x = (xs.max()+xs.min()) / 2
    mid_y = (ys.max()+ys.min()) / 2
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(0, max_range)

    # Plot image planes for each camera.
    for index, pos in enumerate(camera_positions):
        X, Y, Z = get_image_plane(camera_orientations[index][0], camera_orientations[index][1], camera_orientations[index][2], xstride=CAMERA_RESOLUTION[0], ystride=CAMERA_RESOLUTION[1])
        X *= max_range/FOCAL_LEN / 8
        Y *= max_range/FOCAL_LEN / 8
        Z *= max_range/FOCAL_LEN / 8
        X += pos[0]
        Y += pos[1]
        Z += pos[2]

        ax.plot_surface(X, Y, Z, facecolors=np.array([[[1, 1, 0.25] for y in x] for x in Z]))
        ax.plot([pos[0], X[0][0]], [pos[1], Y[0][0]], [pos[2], Z[0][0]], color='black')
        ax.plot([pos[0], X[-1][0]], [pos[1], Y[-1][0]], [pos[2], Z[-1][0]], color='black')
        ax.plot([pos[0], X[-1][-1]], [pos[1], Y[-1][-1]], [pos[2], Z[-1][-1]], color='black')
        ax.plot([pos[0], X[0][-1]], [pos[1], Y[0][-1]], [pos[2], Z[0][-1]], color='black')

    # Display from multiple angles.
    if animated:
        for angle in range(0, 360, 15):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(1)
    else:
        ax.view_init(30, 30)
        plt.show()

if __name__ == "__main__":
    camera_positions = np.array([[-10, -10, 10], [10, 10, 10], [-10, 10, 10], [10, -10, 10]])
    camera_orientations = np.array([[45, -45, 30], [45, 135, 10], [45, -135, -20], [45, 45, 0]])
    camera_names = ['6719', '6720', '6721', '6722']

    plot_cameras(camera_positions, camera_orientations, camera_names, animated=True)
