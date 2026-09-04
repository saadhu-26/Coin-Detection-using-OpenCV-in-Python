# Coin-Detection-using-OpenCV-in-Python
# Aim
This project detects and counts coins in an image using Python and OpenCV. It uses the Hough Circle Transform to identify circular coins.

# Technologies Used:
- Python
- OpenCV
- NumPy

# How It Works:
- Read the input image.
- Convert the image to grayscale.
- Apply Gaussian Blur.
- Detect circles using cv2.HoughCircles().
- Draw circles around detected coins.
- Count the detected coins.

# Program
```
import cv2
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
image=cv2.imread('Coin.png')
imageCopy = image.copy()
plt.imshow(image[:,:,::-1]);
plt.title("Original Image")
plt.show()
imageGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(12,12))
plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image")
plt.subplot(122); plt.imshow(imageGray,cmap='gray');plt.title("Grayscale Image")
imageB,imageG,imageR=cv2.split(imageCopy)
plt.figure(figsize=(20,12))
plt.subplot(141);plt.imshow(image[:,:,::-1]);plt.title("Original Image")
plt.subplot(142);plt.imshow(imageB,cmap='gray');plt.title("Blue Channel")
plt.subplot(143);plt.imshow(imageG,cmap='gray');plt.title("Green Channel")
plt.subplot(144);plt.imshow(imageR,cmap='gray');plt.title("Red Channel");
plt.show()
thresh=20
maxValue=255
th,dst_bin=cv2.threshold(imageG,thresh,maxValue,cv2.THRESH_BINARY)
th,dst_bin_inv=cv2.threshold(imageG,thresh,maxValue,cv2.THRESH_BINARY_INV)
plt.imshow(dst_bin_inv,cmap='gray')
plt.title('Threshold Binary Inverse')
plt.axis('off')
plt.show()
kernel=np.ones((7,7),dtype=np.uint8)
kernel
dst=cv2.dilate(dst_bin_inv,kernel)
plt.imshow(dst,cmap='gray');plt.title('Dilated Image ');plt.show()
plt.imshow(dst,cmap='gray');plt.title('Dilated Image Iteration 2');plt.show()
plt.figure(figsize=(15, 10))
plt.subplot(231)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")
plt.axis("off")
plt.subplot(232)
plt.imshow(imageGray, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")
plt.subplot(233)
plt.imshow(dst_bin_inv, cmap='gray')
plt.title("Thresholded Image")
plt.axis("off")
plt.subplot(234)
plt.imshow(dst, cmap='gray')
plt.title("Dilated Image")
plt.axis("off")
plt.subplot(235)
plt.imshow(dst, cmap='gray')
plt.title("Dilated Image Iteration 2")
plt.axis("off")
plt.show()
ksize=(7,7)
k1=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,ksize)
erod=cv2.erode(dst,k1)
plt.imshow(erod,cmap='gray');plt.title("Eroded Image");plt.show()
params = cv2.SimpleBlobDetector_Params()
params.blobColor = 0
params.minDistBetweenBlobs = 2
# Filter by Area.
params.filterByArea = False
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8
params.filterByInertia =True
params.minInertiaRatio = 0.8
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(erod)
print(f"Number of coins detected: {len(keypoints)}")
finalImage = image.copy()
# Draw circles and centers
for keypoint in keypoints:
 x, y = keypoint.pt
 radius = keypoint.size / 2
 # Green circle
 cv2.circle(
 finalImage,
 (int(x), int(y)),
 int(radius),
 (0, 255, 0),
 2
 )
 # Blue center
 cv2.circle(
 finalImage,
 (int(x), int(y)),
 4,
 (255, 0, 0),
 -1
 )
finalImageRGB = cv2.cvtColor(finalImage, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 8))
plt.imshow(finalImageRGB)
plt.title("Final Image")
plt.axis("off")
plt.show()
```

# Output

<img width="363" height="349" alt="image" src="https://github.com/user-attachments/assets/a74d3ff7-0b4b-48ac-91eb-8e5afaea1e7b" />

<img width="830" height="450" alt="image" src="https://github.com/user-attachments/assets/b11b03ae-5437-4be5-b8b1-5871a2c3c8bf" />

<img width="900" height="242" alt="image" src="https://github.com/user-attachments/assets/4461dca9-7f71-4b06-8f7e-9f75d3bd4654" />

<img width="335" height="327" alt="image" src="https://github.com/user-attachments/assets/b5821b55-48a8-4743-bb7e-67fd26b146c3" />

<img width="357" height="354" alt="image" src="https://github.com/user-attachments/assets/92d91e5e-1790-4854-99c7-01581fffa42f" />

<img width="506" height="539" alt="image" src="https://github.com/user-attachments/assets/9237ac8a-6e8b-4f83-9048-565f38659cbd" />

# Result
This project demonstrates the use of OpenCV and Hough Circle Transform for automatic coin detection and counting. It is a simple and effective application of Digital Image Processing.
