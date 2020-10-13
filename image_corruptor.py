"""
Takes the image called 'img.png' in the current directory and applies some distortion to the 3 channels.
The image corrupted is shown (together with the original one) and is saved as 'im2.png'.
"""

import matplotlib.pyplot as plt
import numpy as np

im = plt.imread('img.png')
r = im[:, :, 0]
g = im[:, :, 1]
b = im[:, :, 2]
x = np.empty((256, 256, 3))
x[:, :, 0] = r + 0.5
x[:, :, 1] = g + 0.5
x[:, :, 2] = b
x = x.clip(0, 1)

print(np.sum(im[:, :, 0] - x[:, :, 0]))
print('\n')
print(np.sum(im[:, :, 1] - x[:, :, 1]))
print('\n')
print(np.sum(im[:, :, 2] - x[:, :, 2]))

plt.figure(1)
plt.imshow(im)

plt.figure(2)
plt.imshow(x)
plt.imsave('im2.png', x)
plt.show()


