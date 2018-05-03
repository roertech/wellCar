# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from scipy import stats


#image = mpimg.imread('test_images/solidWhiteRight.jpg')
"""
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image
"""


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
       
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape,3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img, lines

    # Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, a=0.8, b=1.0, r=0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, r)

"""
image = mpimg.imread('test_images/solidWhiteCurve.jpg')
gray_image = grayscale(image)
plt.imshow(gray_image, cmap='gray')
plt.show()

blur_img = gaussian_blur(gray_image, 7)
canny_img = canny(blur_img, 40, 120)
"""

"""
plt.imshow(canny_img, cmap='gray')
plt.show()
"""

"""
mask = np.array([[[100,540], [480,315], [960,540], [700,540], [480,350], [300,540]]])
masked_img = region_of_interest(canny_img, mask)

plt.imshow(masked_img, cmap='gray')

plt.show()


rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 1     # minimum number of votes (intersections in Hough grid cell)
min_line_len = 3  #minimum number of pixels making up a line
max_line_gap = 5  # maximum gap in pixels between connectable line segments
hough_img, lines = hough_lines(masked_img, rho, theta, threshold, min_line_len, max_line_gap)

plt.imshow(hough_img, cmap='gray')

plt.show()

raw_img = weighted_img(hough_img, image)
plt.imshow(raw_img)
plt.show()
"""
def f(x, a, b):  
    return a + b*x  

def show(x,y,xdata=0,slope=0,intercept=0):
    #xdata = np.linspace(1, 5, 20)  
    plt.grid(True)  
    plt.xlabel('x axis')    
    plt.ylabel('y axis')   
    #plt.text(2.5, 4.0, r'$y = ' + str(intercept) + ' + ' + str(slope) +'*x$', fontsize=18)  
    #plt.plot(xdata, f(xdata, intercept,slope), 'b', linewidth=1)  
    plt.plot(x,y,'ro')  
    plt.show() 
LINE_REJECT_DEGREES=10
def get_separated_coords(lines):
    left_lines_coords = []
    right_lines_coords = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            dx=x2-x1
            dy=y2-y1
            angle=math.atan2(dy, dx)*180/math.pi
            
            if abs(angle)>LINE_REJECT_DEGREES:
                print(abs(angle))
                right_most_point = max(x1,x2)
                if right_most_point < 480:
                    left_lines_coords.append([x1, y1])
                    left_lines_coords.append([x2, y2])
                else:
                    right_lines_coords.append([x1, y1])
                    right_lines_coords.append([x2, y2])
    left_lines_coords = np.array(left_lines_coords)
    right_lines_coords = np.array(right_lines_coords)    
    return left_lines_coords, right_lines_coords


def get_line(coords, y1=540, y2=330):
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(coords))
    print(slope)
    print(intercept)
    x1 = int(np.round((y1 - intercept)/slope))
    x2 = int(np.round((y2 - intercept)/slope))    
    line = [[x1,y1,x2,y2]]
    return line


def draw_lanes(image, lines):
    best_left_coords, best_right_coords = get_separated_coords(lines)
    print(best_left_coords)
    x=best_right_coords[:,0]
    print(x)
    y=best_right_coords[:,1]
    show(x,y)
    #from sklearn.cluster import KMeans
    X=best_right_coords
    #y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
    #plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    #plt.show()
    left_line = get_line(best_left_coords)
    right_line = get_line(best_right_coords)
    print(right_line)
    draw_lines(image, [left_line], color=[0, 0, 255], thickness=10)
    draw_lines(image, [right_line], color=[0, 255, 0], thickness=10)
    return image
"""
image = draw_lanes(image, lines)
plt.imshow(image)
plt.show()
"""

 

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    image=cv2.resize(image, (960, 540))
    gray_image = grayscale(image)
    blur_img = gaussian_blur(gray_image, 7)
    canny_img = canny(blur_img, 40, 120)
    mask = np.array([[[100,540], [200,315], [960,315], [700,540], [480,350], [300,540]]])
    masked_img = region_of_interest(canny_img, mask)
    plt.imshow(masked_img)
    plt.show()
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 1     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 3  #minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    hough_img, lines = hough_lines(masked_img, rho, theta, threshold, min_line_len, max_line_gap)
    image = draw_lanes(image, lines)
    return image

#image = mpimg.imread('test_images/solidWhiteCurve.jpg')
#image = mpimg.imread('test_images/solidWhiteRight.jpg')
#image = mpimg.imread('test_images/solidYellowCurve.jpg')
#image = mpimg.imread('test_images/solidYellowCurve2.jpg')
image = mpimg.imread('../test_images/tu.jpg')
image=process_image(image)
plt.imshow(image)
plt.show()


x = [3.5, 2.5, 4.0, 3.8, 2.8, 1.9, 3.2, 3.7, 2.7, 3.3]   #高中平均成绩  
y = [3.3, 2.2, 3.5, 2.7, 3.5, 2.0, 3.1, 3.4, 1.9, 3.7]   #大学平均成绩  
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)  
slope = round(slope,3)  
intercept = round(intercept,3)  
print (slope, intercept) 
  


