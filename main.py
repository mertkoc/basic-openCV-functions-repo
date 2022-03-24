"""
Author: Mert Koc

Refined: 24/03/2022

"""

# I have a shutterstock video for some parts:
# ref: https://www.shutterstock.com/video/clip-11921927-pick-grilled-salmon-dish-olives-baguette

# import argparse
import time
import cv2
# import sys
import random
import numpy as np

COUNT = 4
DOUBLE_COUNT = COUNT * 2
OFFSET = 10
ORANGE_MIN_RGB_vals = (0, 50, 25)
ORANGE_MAX_RGB_vals = (120, 200, 255)
ORANGE_MIN_HSV_vals = (7, 80, 100)
ORANGE_MAX_HSV_vals = (15, 225, 255)
CLOSING_KERNEL_SIZE = 5
ORANGE_MIN_RGB = np.array(ORANGE_MIN_RGB_vals, np.uint8)
ORANGE_MAX_RGB = np.array(ORANGE_MAX_RGB_vals, np.uint8)
ORANGE_MIN_HSV = np.array(ORANGE_MIN_HSV_vals, np.uint8)
ORANGE_MAX_HSV = np.array(ORANGE_MAX_HSV_vals, np.uint8)


def black_and_white(time_first, frame):
    time_check = time.time() - time_first

    if time_check >= 0.8:  # Every 0.5 secs
        time_first = time.time()
        is_gray = random_switch(frame)

    if is_gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.putText(img=frame, text='Black&White', org=(450, 700), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2,
                    color=(0, 0, 255), thickness=3)

    else:
        cv2.putText(img=frame, text='Colored', org=(450, 700), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2,
                    color=(0, 0, 255), thickness=3)

    return frame, time_first


def gaussian_blur(time_first, frame, kernel_size):
    time_check = time.time() - time_first
    if time_check >= 0.6:  # Every 0.5 secs
        time_first = time.time()
        kernel_size += 2
    frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    cv2.putText(img=frame, text=f"Gaussian {kernel_size}x{kernel_size}", org=(450, 700),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(0, 0, 255), thickness=3)

    return frame, time_first, kernel_size


def bilateral_blur(time_first, frame, filter_size, sigma):
    time_check = time.time() - time_first
    if time_check >= 2:  # Every 0.5 secs
        time_first = time.time()
        filter_size += 1
        sigma += 75
    frame = cv2.bilateralFilter(frame, int(filter_size), 0, sigma)
    cv2.putText(img=frame, text=f"Bilateral filter={int(filter_size)}, sigma={sigma}", org=(100, 700),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(0, 0, 255), thickness=3)

    return time_first, frame, filter_size, sigma


def random_switch():
    if random.random() >= 0.5:
        return True
    return False


def gaussian_and_bilateral():
    vid = cv2.VideoCapture('kids.mp4')
    fps = vid.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"fps = {fps}, duration {duration}")
    i = 1

    if not vid.isOpened():
        print("Error opening video stream or file")

    while vid.isOpened():
        ret, frame = vid.read()

        if not ret:
            break

        dim = (1280, 720)

        # resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        dim2 = (dim[0] // 2, dim[1] // 2)
        frame_gauss = cv2.resize(frame, dim2, interpolation=cv2.INTER_AREA)
        frame_bilateral = cv2.resize(frame, dim2, interpolation=cv2.INTER_AREA)
        dim3 = (dim[0] // 2, dim[1])
        frame = cv2.resize(frame, dim3, interpolation=cv2.INTER_AREA)

        if i >= 50:
            kernel_size = 25
            range = 25
            sigma = 150
            frame_gauss = cv2.GaussianBlur(frame_gauss, (kernel_size, kernel_size), 0)
            frame_bilateral = cv2.bilateralFilter(frame_bilateral, range, sigma, sigma)
            cv2.putText(img=frame_gauss, text=f"Gaussian {kernel_size}x{kernel_size}", org=(200, 30),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
            cv2.putText(img=frame_bilateral, text=f"Bilateral filter range={range},sigma={sigma}", org=(5, 30),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
        else:
            range = 15
            sigma = 150
            kernel_size = 15
            frame_gauss = cv2.GaussianBlur(frame_gauss, (kernel_size, kernel_size), 0)
            frame_bilateral = cv2.bilateralFilter(frame_bilateral, range, sigma, sigma)
            cv2.putText(img=frame_gauss, text=f"Gaussian {kernel_size}x{kernel_size}", org=(200, 30),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
            cv2.putText(img=frame_bilateral, text=f"Bilateral filter range={range},sigma={sigma}", org=(5, 30),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)

        cv2.putText(img=frame, text=f"No filter", org=(260, 30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                    color=(0, 0, 255), thickness=1)

        frame_1 = cv2.vconcat([frame_gauss, frame_bilateral])
        frame = cv2.hconcat([frame, frame_1])

        cv2.putText(img=frame, text=f"Larger sigma smoothes the larger features (texture is almost gone),",
                    org=(30, 680), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0), thickness=1)
        cv2.putText(img=frame, text=f"Bilateral filter approaches to Gaussian filter with larger range", org=(60, 710),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0), thickness=1)
        filename = f"out/IMG{i:04d}.jpg"
        cv2.imwrite(filename=filename, img=frame)
        i += 1
        cv2.imshow('frame', frame)
        wait_key = cv2.waitKey(1) & 0xFF # Necessary for imshow function, you can disable this if you won't use imshow
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def filter_foreground(frame, mode, kernel_size=CLOSING_KERNEL_SIZE):
    # frame = frame.copy()

    if mode == "HSV":
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # create a mask using the bounds set
        mask = cv2.inRange(frameHSV, ORANGE_MIN_HSV, ORANGE_MAX_HSV)

        # Filter only the red colour from the original image using the mask(foreground)
        res = cv2.bitwise_and(frameHSV, frameHSV, mask=mask)
        res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)

    elif mode == "HSV+morph":
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # create a mask using the bounds set
        mask = cv2.inRange(frameHSV, ORANGE_MIN_HSV, ORANGE_MAX_HSV)

        # Filter only the red colour from the original image using the mask(foreground)
        res = cv2.bitwise_and(frameHSV, frameHSV, mask=mask)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
        res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)

    elif mode == "RGB":
        # frameHSV = cv2.cvtColor(frame, cv2.CV_BGR2HSV)
        # create a mask using the bounds set
        mask = cv2.inRange(frame, ORANGE_MIN_RGB, ORANGE_MAX_RGB)
        res = cv2.bitwise_and(frame, frame, mask=mask)

    elif mode == "RGB+morph":
        # frameHSV = cv2.cvtColor(frame, cv2.CV_BGR2HSV)
        # create a mask using the bounds set
        mask = cv2.inRange(frame, ORANGE_MIN_RGB, ORANGE_MAX_RGB)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        # res = cv2.bitwise_and(frame, frame)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

    return res


def grab_them_fish():
    vid = cv2.VideoCapture('kids.mp4')
    fps = vid.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"fps = {fps}, duration {duration}")
    i = 1
    # sigma = 8
    if vid.isOpened() == False:
        print("Error opening video stream or file")

    while vid.isOpened():
        ret, frame = vid.read()

        # print(ret)
        if not ret:
            break

        dim = (1280, 720)
        dim2 = (dim[0] // 2, dim[1] // 2)

        frame = cv2.resize(frame, dim2, interpolation=cv2.INTER_AREA)

        CLOSING_KERNEL_SIZE = 9
        frame_hsv_grab = filter_foreground(frame, mode="HSV")
        frame_hsv_morph_grab = filter_foreground(frame, mode="HSV+morph", kernel_size=CLOSING_KERNEL_SIZE)
        frame_rgb_grab = filter_foreground(frame, mode="RGB")
        frame_rgb_morph_grab = filter_foreground(frame, mode="RGB+morph", kernel_size=CLOSING_KERNEL_SIZE)

        cv2.putText(img=frame_hsv_grab, text=f"HSV Grab", org=(30, 30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                    color=(0, 0, 255), thickness=1)
        cv2.putText(img=frame_hsv_grab, text=f"R:[{ORANGE_MIN_HSV_vals}, {ORANGE_MAX_HSV_vals}]", org=(5, 60),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
        cv2.putText(img=frame_hsv_morph_grab, text=f"HSV+Morph(closing) Grab", org=(5, 30),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
        cv2.putText(img=frame_hsv_morph_grab, text=f"R:[{ORANGE_MIN_HSV_vals}, {ORANGE_MAX_HSV_vals}]", org=(5, 60),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
        cv2.putText(img=frame_hsv_morph_grab, text=f"closing kernel size {CLOSING_KERNEL_SIZE}", org=(5, 90),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
        cv2.putText(img=frame_rgb_grab, text=f"RGB Grab", org=(5, 30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                    color=(0, 0, 255), thickness=1)
        cv2.putText(img=frame_rgb_grab, text=f"R:[{ORANGE_MIN_RGB_vals}, {ORANGE_MAX_RGB_vals}]", org=(5, 60),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
        cv2.putText(img=frame_rgb_morph_grab, text=f"RGB+Morph(closing) Grab", org=(5, 30),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
        cv2.putText(img=frame_rgb_morph_grab, text=f"R:[{ORANGE_MIN_RGB_vals}, {ORANGE_MAX_RGB_vals}]", org=(5, 60),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
        cv2.putText(img=frame_rgb_morph_grab, text=f"closing kernel size {CLOSING_KERNEL_SIZE}", org=(5, 90),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)

        frame_1 = cv2.vconcat([frame_hsv_grab, frame_hsv_morph_grab])
        frame_2 = cv2.vconcat([frame_rgb_grab, frame_rgb_morph_grab])

        # print(frame_1.shape)
        # print(frame_2.shape)
        # print(frame_1.shape)
        frame = cv2.hconcat([frame_1, frame_2])

        cv2.putText(img=frame, text=f"It can be seen that RGB mask brings more false-positives", org=(50, 590),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255), thickness=1)
        cv2.putText(img=frame, text=f"to get a similar performance compared to HSV mask. The reason that",
                    org=(10, 620), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255), thickness=1)
        cv2.putText(img=frame, text=f"hue is very correlated to color, and uncorrelated to light-intensity",
                    org=(10, 650), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255), thickness=1)
        cv2.putText(img=frame, text=f"Morph closing is very suitable as it fills the holes", org=(10, 680),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255), thickness=1)
        cv2.putText(img=frame, text=f"larger kernel size fills more holes", org=(10, 710),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255), thickness=1)

        filename = f"out/IMG{i:04d}.jpg"
        cv2.imwrite(filename=filename, img=frame)
        i += 1
        cv2.imshow('frame', frame)
        wait_key = cv2.waitKey(1) & 0xFF
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def switch_colored_bw() -> None:
    # define a video capture object
    vid = cv2.VideoCapture(0)

    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width, height)

    recording = False

    time_first = time.time()
    count_down = COUNT
    # is_gray = False
    start_time = time.time()
    i = 1
    # kernel_size = 3
    filter_size = 3
    sigma = 80
    while True:
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        filename = f"out/IMG{i:04d}.jpg"
        cv2.imwrite(filename=filename, img=frame)
        i += 1

        if recording:
            elapsed_time = time.time() - start_time
            if elapsed_time >= count_down + OFFSET:
                break

            time_first, frame, filter_size, sigma = bilateral_blur(time_first, frame, filter_size, sigma)
            filename = f"out/IMG{i:04d}.jpg"
            cv2.imwrite(filename=filename, img=frame)
            i += 1

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        wait_key = cv2.waitKey(1) & 0xFF
        if wait_key == ord('q') or wait_key == ord(chr(27)):
            break
        elif wait_key == ord('r'):
            recording = ~recording
            if recording:
                time_first = time.time()
                count_down = COUNT
                start_time = time.time()
                i = 1
                filter_size = 5
                sigma = 75

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def in_between_scenes():
    frame = np.zeros((720, 1280), dtype=np.uint8)
    cv2.putText(frame, "Basic image processing", org=(40, 370), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=3,
                color=(255, 255, 255), thickness=3)
    for i in range(1, 101):
        filename = f"out/IMG{i:04d}.jpg"
        cv2.imwrite(filename=filename, img=frame)


def sobel_filter():
    # define a video capture object
    vid = cv2.VideoCapture(0)

    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width, height)
    vid.set(cv2.CAP_PROP_FPS, 25)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    kernel_size = 5
    for i in range(1, 126):  # 5 secs with 25 fps
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        dim = (1280, 720)
        dim2 = (dim[0] // 2, dim[1] // 2)
        frame = cv2.resize(frame, dim2, interpolation=cv2.INTER_AREA)
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

        # Sobel Edge Detection
        if (i - 1) != 0 and (i - 1) % 25 == 0:
            kernel_size += 2

        sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_8U, dx=1, dy=0,
                           ksize=kernel_size)  # Sobel Edge Detection on the X axis

        sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_8U, dx=0, dy=1,
                           ksize=kernel_size)  # Sobel Edge Detection on the Y axis

        sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_8U, dx=1, dy=1,
                            ksize=kernel_size)  # Combined X and Y Sobel Edge Detection

        dummy = np.zeros((dim2[1], dim2[0]), dtype=np.uint8)

        img_gray = np.stack((img_gray,) * 3, axis=-1)
        sobelx = np.stack((sobelx, dummy, dummy), axis=-1)  # X edges are blue
        sobely = np.stack((dummy, sobely, dummy), axis=-1)
        sobelxy = np.stack((sobelxy,) * 3, axis=-1)

        cv2.putText(img=sobelx, text=f"Sobel X kernel size={kernel_size}", org=(160, 30),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
        cv2.putText(img=sobely, text=f"Sobel Y kernel size={kernel_size}", org=(160, 30),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
        cv2.putText(img=sobelxy, text=f"Sobel XY kernel size={kernel_size}", org=(160, 30),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
        cv2.putText(img=img_gray, text=f"Grayscale Image", org=(170, 30), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=1)

        frame_2 = cv2.vconcat([sobelx, sobely])
        frame_1 = cv2.vconcat([img_gray, sobelxy])

        frame = cv2.hconcat([frame_1, frame_2])

        cv2.putText(img=frame, text=f"Larger kernel size intensifies the thickness of the edges", org=(25, 700),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.25, color=(0, 0, 255), thickness=2)

        cv2.imshow('frame', frame)
        wait_key = cv2.waitKey(1) & 0xFF

        if wait_key == ord('q') or wait_key == ord(chr(27)):
            break

        filename = f"out/IMG{i:04d}.jpg"
        cv2.imwrite(filename=filename, img=frame)

    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def detect_circles():
    # define a video capture object
    vid = cv2.VideoCapture(0)

    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width, height)
    vid.set(cv2.CAP_PROP_FPS, 25)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    for i in range(1, 251):  # 10 secs with 25 fps
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        dim = (1280, 720)
        dim2 = (dim[0] // 2, dim[1] // 2)
        frame = cv2.resize(frame, dim2, interpolation=cv2.INTER_AREA)
        frame_2 = frame.copy()
        frame_3 = frame.copy()
        frame_4 = frame.copy()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)

        frames = [frame, frame_2, frame_3, frame_4]

        dp = [1, 2, 3, 4]
        min_dist = [10, 20, 30, dim2[1] / 4]
        param1 = [20, 50, 100, 150]
        param2 = [100, 150, 200, 250]

        param_1 = param1[(i - 1) // 63]

        for idx, val1 in enumerate(param2):
            dP = dp[1]
            minDist = min_dist[0]

            param_2 = val1
            maxRad = 200
            circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=dP, minDist=minDist,
                                       param1=param_1, param2=param_2, minRadius=0, maxRadius=maxRad)

            # circles = cv2.HoughCircles(img_blur,cv2.HOUGH_GRADIENT,1.2,100)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circ in circles[0, :]:
                    # draw the outer circle
                    cv2.circle(frames[idx], (circ[0], circ[1]), circ[2], (0, 255, 255), 3)

            cv2.putText(img=frames[idx], text=f"dp={dP},minDist={minDist},param1={param_1},", org=(5, 30),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0), thickness=1)
            cv2.putText(img=frames[idx], text=f"param2={param_2},maxRad={maxRad}", org=(5, 60),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0), thickness=1)

        frame_1 = cv2.vconcat([frame, frame_2])
        frame_2 = cv2.vconcat([frame_3, frame_4])

        frame = cv2.hconcat([frame_1, frame_2])

        cv2.putText(img=frame, text=f"Param1 is the upper threshold val. of Canny filter inside", org=(25, 620),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.25, color=(0, 0, 0), thickness=2)
        cv2.putText(img=frame, text=f"Hough Trans., affecting number of accepted edges, param2", org=(25, 650),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.25, color=(0, 0, 0), thickness=2)
        cv2.putText(img=frame, text=f"is accumulator threshold, which gives higher false-positive", org=(25, 680),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.25, color=(0, 0, 0), thickness=2)
        cv2.putText(img=frame, text=f"rate for the lower values of the threshold", org=(25, 710),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.25, color=(0, 0, 0), thickness=2)

        cv2.imshow('frame', frame)
        wait_key = cv2.waitKey(1) & 0xFF

        if wait_key == ord('q') or wait_key == ord(chr(27)):
            break

        filename = f"out/IMG{i:04d}.jpg"
        cv2.imwrite(filename=filename, img=frame)

    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def track_object():
    # define a video capture object
    vid = cv2.VideoCapture(0)

    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width, height)
    vid.set(cv2.CAP_PROP_FPS, 25)
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    dim = (1280, 720)

    observation = np.ones((dim[1], dim[0]))
    alpha = 1

    for i in range(1, 126):  # 5 secs with 25 fps

        mode = 0 if (i - 1) <= 50 else 1

        ret, frame = vid.read()

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)

        dP = 1
        minDist = dim[1] / 4
        param1 = 50
        param2 = 80
        maxRad = 150

        circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=dP, minDist=minDist,
                                   param1=param1, param2=param2, minRadius=0, maxRadius=maxRad)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circ in circles[0, :]:
                # draw the outer circle
                cv2.rectangle(frame, (circ[0] - circ[2], circ[1] - circ[2]), (circ[0] + circ[2], circ[1] + circ[2]),
                              (0, 255, 255), 3)

                if mode == 1:
                    observation[circ[1] - 1, circ[0] - 1] += 1
                    observation[circ[1] - 1, circ[0]] += 1
                    observation[circ[1] - 1, circ[0] + 1] += 1
                    observation[circ[1], circ[0] - 1] += 1
                    observation[circ[1], circ[0]] += 1
                    observation[circ[1], circ[0] + 1] += 1
                    observation[circ[1] + 1, circ[0] - 1] += 1
                    observation[circ[1] + 1, circ[0]] += 1
                    observation[circ[1] + 1, circ[0] + 1] += 1

        else:
            if mode == 1:
                observation += 1

        if mode == 1:
            prob_dist = (observation + alpha) / (observation.sum() + alpha)
            norm_image = cv2.normalize(prob_dist, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            norm_image = norm_image.astype(np.uint8)
            norm_image = np.stack((norm_image,) * 3, axis=-1)
            alpha = 0.15
            beta = (1.0 - alpha)
            frame = cv2.addWeighted(frame, alpha, norm_image, beta, 0.0, dtype=cv2.CV_8U)

        cv2.imshow('frame', frame)

        wait_key = cv2.waitKey(1) & 0xFF
        if wait_key == ord('q') or wait_key == ord(chr(27)):
            break

        filename = f"out/IMG{i:04d}.jpg"
        cv2.imwrite(filename=filename, img=frame)

    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    switch_colored_bw()
    gaussian_and_bilateral()
    grab_them_fish()
    in_between_scenes()
    sobel_filter()
    detect_circles()
    track_object()
