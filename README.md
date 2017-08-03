# **Finding Lane Lines on the Road**

[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteRight.jpg "Grayscale"

---
## Installation
Run `bash run.sh` and follow instructions on terminal to get to Jupyter Notebook.

## Reflection
The pipeline consisted of the following steps:
```
gray_img = grayscale(img)
gray_img = gaussian_blur(gray_img, 5)
edge_img = canny(gray_img, 50, 150)
h_lines = hough_lines(edge_img, 1, np.pi/180, 10, 30, 10)
h_lines = region_of_interest(h_lines, np.array([[left_bottom, right_bottom, apex]]))
final_img = weighted_img(h_lines, img)
```    
They were initially converted to a grayscale followed by blurring the image. The blurring was done so that unnecessary details of the image such as, texture of the road and spurious lines would be blurred. This made it easier for the canny edge filter to determine the most prominent edges. This was followed by using the hough lines transform to extract which of these edges were lying on straight lines. It is however a valid assumption that the lane lines would lie in an triangular area that encompasses the two bottom corners and the middle of the image. Hence we limited ourselves to all the edges only within this region. Finally these lines were super imposed on the original image in the function `weighted_img`.

In order to draw a single line on the left and right lanes, I modified the `hough_lines()` function by extending all the found lines down to the bottom corner. This was done by using the simple equation of a line and finding the coordinate when `y = image_height`. This is portrayed in the code snippet below.
```
if y2>y1:
    grad = (y1-y2)/(y1-img.shape[0])
    x_star = np.round(x1 - (x1-x2)/grad)
    lines[i] = np.array([x1,y1,x_star,img.shape[0]])
```                    

![alt text][image1]

### Extensions
There were three important modifications that were done in this project.
1. It is safe to assume that the lane lines that are closest to the car would have an angle that is greater than 30 degrees to the horizon. Hence, after the endpoints of the edge was extracted, all lines that are shallower than this angle were discarded.
2. There were a lot of spurious edges initially that tended to start outside the region of interest. This was due to the fact that the `region_of_interest` function discarded edges that were outside the triangular region, while only _truncating_ an edge if it was partially outside the ROI. We extended this by discarding all but the edges that were completely contained within the ROI (i.e. the endpoints were within the ROI).
3. The x coordinate of the line extended to the bottom as well as its gradients were clustered using Kmeans (for 2 clusters). The cluster centers were taken to indicate the starting x coordinate and the gradient of the two lane lines. If the Kmean centers were less than 100 pixels apart, those lines were averaged to create 1 cluster instead.
4. With videos however, there were occasions when the second lane marking was missing. In these occasions the previous known x coordinate and gradient was used. Furthermore, to reduce sudden fluctuations of lane lines, we employed exponential smoothing of the cluster centers, such that: `new_center = 0.9 x current_center + 0.1 x previous_center`. The following code snippet portrays this concept.
```
    if len(c_hist)>1: #If its not the first frame
        if clusters.shape[0]==2:
            clusters = 0.9*clusters +0.1*c_hist[-1]
        elif len(clusters)==0:
            clusters = c_hist[-1]
```

<iframe width="560" height="315" src="https://www.youtube.com/embed/IqnMC2vrkQs" frameborder="0" allowfullscreen></iframe>

## Shortcomings

Currently it is quite vulnerable to light conditions. This can be seen in the optional video where the left lane line can be spurious since it depends on previous frames.

## Futurework and Improvements
The parameters were hand tuned to the images that were shown here. It is questionable if it would perform in a unseen test set. Hence it would be useful to use Deep Learning to find the lanes themselves.
