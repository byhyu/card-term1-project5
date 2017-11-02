import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
# image generation for writeup

# # Read in images of cars and notcars
car_imgs = glob.glob('vehicles/*/*.png')
notcar_imgs =glob.glob('non-vehicles/*/*.png')
n_car_imgs = len(car_imgs)
n_notcar_imgs = len(notcar_imgs)
print(n_car_imgs)
print(n_notcar_imgs)

fig=plt.figure(figsize=(10.,5))
for i in range(3):
    car_img = cv2.imread(car_imgs[randint(n_car_imgs)])
    notcar_img = cv2.imread(notcar_imgs[randint(n_notcar_imgs)])
    plt.subplot(3,2,i*2+1)
    plt.imshow(car_img)
    plt.subplot(3,2,i*2+2)
    plt.imshow(notcar_img)

plt.tight_layout()
fig.savefig('output_imgs/car_notcar.png')
plt.show()

##