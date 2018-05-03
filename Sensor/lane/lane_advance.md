高级寻路



The goals / steps of this project are the following:

* 计算摄像头校准矩阵，获取畸形系数Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* 纠正畸形的图片
* 使用颜色变换和梯度等创建二值化图片
* 透视转换变成鸟瞰图 ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chessboard.png "Chessboard"
[image2]: ./output_images/distortion.png "Distortion correction"
[image3]: ./output_images/gradient.png "Gradient threshold combination"
[image4]: ./output_images/color.png "L-S channels threshold combination"
[image5]: ./output_images/combined.png "Combined gradient and color filters"
[image6]: ./output_images/perspective.png "Perspective transformation"
[image7]: ./output_images/sliding.png "Lines found using the sliding window method"
[image8]: ./output_images/easy.png "Lines found using the easy fit method"
[image9]: ./output_images/final.png "After full pipeline is applied"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation. My code is implemented in the accompanying Jupyter notebook `p3_avanced_lanes_detector.ipynb` as a step-by-step guide through all the process, and then all those steps are used in a final function that applies the whole pipeline to each frame of a given video. 

---

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in cells 2 to 3 of the IPython notebook.

The camera is calibrated using the chessboard images provided. Assuming that the chessboard is always at z=0, the object points are always the same for each image. Thus, the object points are generated before looping through each image and a copy is attached for each iteration. For each image under the `camera_cal` folder, the function `cv2.findChessboard` finds the image points of all the internal corners (9 x 6) of the chessboard. Some images do not show all these corners, but the function returns False for these cases, so it is easy to just discard them.

Once all the object and image points are collected, the function `cv2.calibrateCamera` is used to obtain the camera calibration and distortion coefficients, that are passed to the `cv2.undistort`to correct the pictures:

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
Using the calibration and distortion coefficients obtained in the previous step, and the `cv2.undistort` function, images distortion can be corrected:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image.

The applied gradient thresholds (x, y, magnitude and direction) are shown in cells 5 to 8. I found that the combination suggested in the lesson works well, although I changed the threshold values of each individual filter. Example result:

![alt text][image3]

Notice that the clear section of the asphalt makes part of the yellow line invisible to this combination of filters.

As for the color filters (cells 9 and 10), the original undistorted image was converted to HLS color space. A threshold S-channel filter picks up both lines pretty well, but it also picks up the shades of the trees, so it was combined with a threshold L-channel filter that discards low luminosity pixels. Only the pixels that pass both thresholds are shown:

![alt text][image4]

Finally, both gradient and color filter combinations are joined together (cell 11):

![alt text][image5]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective transformation is in cell 12. I used the source and destination points suggested in the example write-up:

	src = np.float32([[585, 460], [203, 720], [1127, 720], [695, 460]])
	dst = np.float32([[320, 0], [320, 720], [960,720], [960, 0]])
 
The code makes use of the `cv2.getPerspectiveTransform` function and the source and destination points to obtain a perspective transformation matrix (M) that can be used in conjunction with the function `cv2.warpPerspective` to obtain a bird-eye view of the image. An inverse perspective transformation matrix (Minv) is also obtained by passing the source points as destination points and vice versa, in order to warp images back to their original perspective when needed:

![alt text][image6]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to detect and fit the lines, I used both methods described in the lesson.

##### Sliding window method (cells 14 to 16)

1. Base positions for the left and right lines are calculated based on the biggest peaks (left and right halves) of the lower half image histogram.
2. Two windows of predetermined height and width are drawn in the lowest part of the image (maximum y coordinate), with their corresponding base position in the middle.
3. For each window, if there are enough non-zero pixels within the window, the x-coordinates of these pixels are averaged, and the resulting value used as base position for the next window.
4. The next pair of windows are drawn based on the averaged positions calculated in the previous step, or the previous positions if not enough non-zero pixels were found.
5. Once all the windows are drawn, a second-degree polynomial is fitted for each line using the coordinates of the non-zero pixels found within the corresponding windows.

![alt text][image7]

##### Easy fit method (cells 17 to 18)

Once a good fit is found for both lines, it is easier to detect and fit the lines for the next frame:

1. For both polynomials from a previous image, all the non-zero pixel coordinates found within a  fixed distance margin are saved.
2. The two polynomials of the current image are fitted using these coordinates, like in the final step of the sliding window method.

![alt text][image8]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.


Both calculations are in cell 19. 

##### Curvature

The method and conversions suggested in the lesson are used:

1. Using the plotted polynomials, two new polynomials are fitted, but this time a pixel space to meters conversion is applied to the pixel coordinates used.
2. The curvature radii of both lines are calculated for the lowest part of the image (maximum y coordinate) using the formula specified in the lesson.
3. The final curvature of the lane is the average of the radii. This step is applied in the video pipeline, since I chose the function to return the curvature radius of both lines.

##### Car position within the lane

The center of the lane is calculated by averaging the x coordinates of the plotted polynomials for the lowest pixels (maximum y coordinate). Assuming that the center of the car is the center of the image, the offset of the car with respect to the lane center is calculated by subtracting the center of the lane from the center of the image, and converting the resulting pixel distance to meters.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Cell 20. The image below shows the result of applying the full pipeline to the example image. The lane area within the detected lane lines is drawn in green. The curvature of the lane and the position of the car with respect to the lane center are shown.

![alt text][image9]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

[Youtube link to my video](https://www.youtube.com/watch?v=6caeJDa7c84 "Video")

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The hardest part of the project was probably finding a good combination of computer vision filters  and their parameters that could make the lane lines stand out for every image, while also hiding other objects from the image that could make line detection very hard, such as shades.
Also, finding a good algorithm for the video pipeline, so that decent results are shown for every frame, is a long trial and error procedure. Smoothing over too many images can cause rigidity in the lane detection process, and is hard to recover from a sequence of difficult frames.

The approach taken works well with the project video, but there are lots of potential problems that are not present in it: glare, faded lane lines, vehicles in the same lane, painted traffic signs, varying lightning conditions and weather conditions, and so on.

The fixed thresholds values used for the different filters cannot possibly work successfully in every situation. A more flexible (intelligent?) approach should be used to identify lane lines.
Another possible improvement is detecting bad fits based on clues like drastic changes in lane curvature or the car position. Right now, the implementation just averages the last ten fits, which makes the lines less wobbly and more robust to bad frames, but cannot recover from long enough sequence of bad frames. Also, what if a hard sequence is present just in the beginning (for example, the first 50 frames)? What reliable reference can be used in such case?