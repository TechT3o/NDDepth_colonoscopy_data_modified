import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the depth map (assume it's a single-channel grayscale image)
depth_map = cv2.imread('Data/C_T1_L1_1_resized/Depth_0000.png', cv2.IMREAD_GRAYSCALE)
depth_map = depth_map.astype(np.float32)  # Convert to float for calculations

# # Compute gradients (depth derivatives)
# dz_dx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
# dz_dy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
#
# # Compute normals
# norm = np.sqrt(dz_dx**2 + dz_dy**2 + 1)
# nx = -dz_dx / norm
# ny = -dz_dy / norm
# nz = 1 / norm
#
# # Combine into a normal map
# normal_map = np.stack((nx, ny, nz), axis=-1)
#
# # Normalize to [0, 1] for visualization
# normal_map_vis = (normal_map + 1) / 2

zy, zx = np.gradient(depth_map)
# You may also consider using Sobel to get a joint Gaussian smoothing and differentation
# to reduce noise
#zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)
#zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)

normal = np.dstack((-zx, -zy, np.ones_like(depth_map)))
n = np.linalg.norm(normal, axis=2)
normal[:, :, 0] /= n
normal[:, :, 1] /= n
normal[:, :, 2] /= n

# offset and rescale values to be in 0-255
normal += 1
normal /= 2
normal *= 255

# Display the normal map
plt.figure(figsize=(10, 10))
plt.imshow(normal[:, :, ::-1].astype(np.uint8))
plt.title("Surface Normal Map")
plt.axis("off")
plt.show()
