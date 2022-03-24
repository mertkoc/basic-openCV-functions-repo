# Basic OpenCV Functions Repo
## TOC
1. [Repo Structure](#repo-structure)
2. [Useful 3rd Party Apps for This Repo](#useful-3rd-party-apps-for-this-repo)
   1. [FFMPEG](#ffmpeg)
3. [Description of the Code](#description-of-the-code)
    1. [switch_colored_bw()](#switch_colored_bw)
    2. [gaussian_and_bilateral()](#gaussian_and_bilateral)
    3. [grab_them_fish()](#grab_them_fish)
    4. [in_between_scenes()](#in_between_scenes)
    5. [sobel_filter()](#sobel_filter)
    6. [detect_circles()](#detect_circles)
    7. [track_object()](#track_object)
---

## Repo Structure
- Example videos are located inside [videos](videos) folder
- [out](out) folder is used by [main.py](main.py) to store the images
- [kids.mp4](kids.mp4) is a video from shutterstock, in which someone 
takes a piece of salmon with their fork from their plate

## Useful 3rd Party Apps for This Repo
### FFMPEG
[FFMPEG](https://ffmpeg.org/) is a great tool for handling videos and images to create your own videos.
Installing it on Linux (like Ubuntu) is quite easy, but installing on Windows
has some extra steps. To install it on Windows, check out wikihow page for
[installing FFmpeg on Windows](https://www.wikihow.com/Install-FFmpeg-on-Windows).

FFMPEG will be very useful to create videos from images and also to stitch videos
to combine them into one video. In [main.py](main.py) line 155:

```
filename = f"out/IMG{i:04d}.jpg"
cv2.imwrite(filename=filename, img=frame)
```

When we set the name of the images this way (which will generate images with 
ordered names, e.g., `IMG0001.jpg`), we can use FFMPEG to generate a video with
frame rate of `25` from the images inside the current folder
with the following line on bash (Linux):

```
ffmpeg -i img%03d.png -r 25 out.mp4 # Example
```

For Windows, it should be:

```
ffmpeg -i img%03d.png -r 25 "../videos/out.mp4"
```

## Description of the Code
The code consist of 7 different methods:
- `swith_colored_bw()`
- `manipulate_video()`
- `grab_them_fish()`
- `in_between_scenes()`
- `sobel_filter()`
- `detect_circles()`
- `track_object()`

After line 603 (`if __name__ == '__main__':`), these methods are placed commented.
You can run them all or one by commenting/uncommenting them.

### switch_colored_bw()

The video switches back and forth random from colored to grayscale. The reason
for using the `cw2.imwrite()` method instead of using the built-in 
`cv2.VideoWriter` class is that the `cv2.VideoWriter` class has a 
parameter `colored` which can be set to `True` or `False`, meaning that
you can only save colored or grayscale videos. 

- Possible work around that I haven't tried: try passing the grayscale image
as 3-channel image by `np.stack((grayscale_image,)*3, axis=-1)` to see if
  `cv2.VideoWriter` class can save both colored and grayscale images.
  
- I used `time` library's `time.time()` method for handling the time, but 
using `cv2.imwrite()` gives the perk of setting exactly how many images you
  want to write which in return you can decide the duration of the video 
  by setting the framerate with FFMPEG. Trying to use `cv2.VideoWriter` class
  with `time.time()` method has the issue of not setting the timing correctly,
  albeit using `cv2.imwrite()` having the issue of missing frames.

### gaussian_and_bilateral()
This method edits the video [kids.mp4](kids.mp4) from 
ShutterStock, 
(link: 
https://www.shutterstock.com/nb/video/clip-11921927-pick-grilled-salmon-dish-olives-baguette
)
to show the difference between bilateral filter and Gaussian filter.

### grab_them_fish()
This method tries to use two color filters (RGB and HSV) to mask images and compare the 
performance of those filters as well the performance when "Closing" morphology
operation is applied to masks. You will see that HSV-based filter will perform
better (will filter more of the image) than the RGB one. This could be explained
due to the fact that "Hue" is a much better filter for colors than individual
RGB values.

### in_between_scenes()
This method creates a black background with text to divide your episodes (if you
ever need it)

### sobel_filter()
This method captures your webcam stream and applies sobel_x, sobel_y, and sobel_xy to 
video stream.

### detect_circles()
This method uses `OpenCV` library's built-in `HoughCircles()` method to obtain
the detected circles' radii and centers. Futhermore, this method captures
your webcam stream.

### track_object()

This method tries to track the object via a "primitive", counting-based, 
probabilistic method. It also blends your webcam stream and the tracking
probabilty.

