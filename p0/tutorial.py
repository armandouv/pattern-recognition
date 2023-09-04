from scipy import misc, ndimage
import matplotlib.pyplot as plt
import numpy as np
import imageio

# Opening and writing to image files
# Writing an array to a file:
f = misc.face()
# misc.imsave('face.png', f) # uses the Image module (PIL)
imageio.imwrite('face.png', f)
plt.imshow(f)
plt.show()

# Creating a numpy array from an image file:
face = misc.face()

# misc.imsave('face.png', face) # First we need to create the PNG file
imageio.imwrite('face.png', face)
face = imageio.imread('face.png')
type(face)
face.shape, face.dtype
#dtype is uint8 for 8-bit images (0-255)

# Opening raw files (camera, 3-D images)
face.tofile('face.raw') # Create raw file
face_from_raw = np.fromfile('face.raw', dtype=np.uint8)
face_from_raw.shape
face_from_raw.shape = (768, 1024, 3)
# Need to know the shape and dtype of the image (how to separate data bytes).
# For large data, use np.memmap for memory mapping:
face_memmap = np.memmap('face.raw', dtype=np.uint8, shape=(768, 1024, 3))
# (data are read from the file, and not loaded into memory)

# Displaying images
# Use matplotlib and imshow to display an image inside a matplotlib figure:
f = misc.face(gray=True) # retrieve a grayscale image
plt.imshow(f, cmap=plt.cm.gray)
plt.show()

# Increase contrast by setting min and max values:
plt.imshow(f, cmap=plt.cm.gray, vmin=30, vmax=200)
# Remove axes and ticks
plt.axis('off')

# Draw contour lines:
plt.contour(f, [50, 200])
plt.show()

# Basic manipulations
# Images are arrays: use the whole numpy machinery.
face = misc.face(gray=True)
face[0, 40]
# Slicing
face[10:13, 20:23]
face[100:120] = 255
lx, ly = face.shape
X, Y = np.ogrid[0:lx, 0:ly]
mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
# Masks
face[mask] = 0
# Fancy indexing
face[range(400), range(400)] = 255

# Statistical information
face = misc.face(gray=True)
face.mean()
face.max(), face.min()

np.histogram
# Geometrical transformations
face = misc.face(gray=True)
lx, ly = face.shape
# Cropping
crop_face = face[lx // 4: - lx // 4, ly // 4: - ly // 4]
# up <-> down flip
flip_ud_face = np.flipud(face)
# rotation
rotate_face = ndimage.rotate(face, 45)
rotate_face_noreshape = ndimage.rotate(face, 45, reshape=False)
