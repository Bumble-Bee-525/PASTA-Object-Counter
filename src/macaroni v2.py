from cv2 import VideoCapture, CAP_DSHOW, FONT_HERSHEY_SIMPLEX, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, namedWindow, waitKey, destroyAllWindows, putText, LINE_AA, line, rectangle, cvtColor, COLOR_BGR2GRAY, Canny, findContours, RETR_EXTERNAL, CHAIN_APPROX_NONE, drawContours, GaussianBlur, contourArea, SimpleBlobDetector_Params, SimpleBlobDetector_create, convertScaleAbs, circle, vconcat, hconcat, COLOR_GRAY2RGB, resize, WINDOW_NORMAL, imwrite, imshow
import numpy as np
from datetime import datetime

print("LAUNCHING. . .")
print("PRODUCED BY BOTCH CO, A SUBSIDIARY OF BRADLEY JAKE NIELSON INDUSTRIES")

# configure settings
camNumber = 0
camWidth = 1280
camHeight = 720
font = FONT_HERSHEY_SIMPLEX
lineColor = (51,204,51)
textColor = (51,204,51)
xMargin = 30
yMargin = 30
xBorder = 300
yBorder = 300
tMin = 50
tMax = 500
alpha = 1.5

# initialize video capture
cap = VideoCapture(camNumber, CAP_DSHOW)
camNumber += 1 

# wait for capture to start
while not cap.isOpened():
    pass

# set video capture resolution
cap.set(CAP_PROP_FRAME_WIDTH, camWidth)
cap.set(CAP_PROP_FRAME_HEIGHT, camHeight)

# initialize default crop settings
targetX = int(camWidth / 2)
targetY = int(camHeight / 2)

'''blob stuf'''
# Setup SimpleBlobDetector parameters.
params = SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200
 
# Filter by Area.
params.filterByArea = True
params.minArea = 150
 
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.0025
 
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.15
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.0025

detector = SimpleBlobDetector_create(params)

# create a window
namedWindow("PASTA", WINDOW_NORMAL)

# run continously
while True:
    #min area of a contour
    minArea = (yMargin * 2) * (xMargin * 2)

    # get an image from webcam
    status, newImage = cap.read()
    processImage = newImage.copy()
    processImage = processImage[targetY - yBorder:targetY + yBorder, targetX - xBorder:targetX + xBorder]

    convertScaleAbs(processImage, processImage, alpha, -30)
    contrastImage = processImage.copy()
    # imshow("contrast", processImage)

    processImage = cvtColor(processImage, COLOR_BGR2GRAY)
    # imshow("Grayed", processImage)

    processImage = GaussianBlur(processImage, (5, 5), 0)
    # imshow("blur", processImage)

    processImage = Canny(processImage, tMin, tMax)
    # imshow("canny", processImage)

    countours, randomShit = findContours(processImage, RETR_EXTERNAL, CHAIN_APPROX_NONE)
    # imshow("contours", processImage)


    #Do blob detection
    blobs = detector.detect(processImage)

    #get only contours with at least minArea
    goodContours = []
    for countour in countours:
        if contourArea(countour) > minArea:
            goodContours.append(countour)

    #concatenate
    #processImage no longer has a color channel, it's just all booleans now, so we give it RGB again
    processImage = cvtColor(processImage, COLOR_GRAY2RGB)

    #stick em together and resize it
    processImage = vconcat([processImage, contrastImage])
    processImage = resize(processImage, (int(camHeight * processImage.shape[1] / processImage.shape[0]), camHeight))

    # display stats
    putText(newImage, f"Camera {camNumber}:", (20, 50), font, 2, textColor, 4, LINE_AA)
    putText(newImage, f"BRADLEY JAKE NEILSON INDUSTRIES", (20, 90), font, 0.75, textColor, 2, LINE_AA)
    putText(newImage, f"BLOB SIZE: {minArea}", (20, 120), font, 0.75, textColor, 2, LINE_AA)
    putText(newImage, f"CEN: {targetX}, {targetY}", (20, 150), font, 0.75, textColor, 2, LINE_AA)
    putText(newImage, f"WID: {xBorder * 2}", (20, 180), font, 0.75, textColor, 2, LINE_AA)
    putText(newImage, f"HEI: {yBorder * 2}", (20, 210), font, 0.75, textColor, 2, LINE_AA)
    putText(newImage, f"RES: {camWidth}, {camHeight}", (20, 240), font, 0.75, textColor, 2, LINE_AA)
    putText(newImage, f"tMIN: {tMin}", (20, 270), font, 0.75, textColor, 2, LINE_AA)
    putText(newImage, f"tMIN: {tMax}", (20, 300), font, 0.75, textColor, 2, LINE_AA)
    putText(newImage, f"ALPHA: {alpha}", (20, 330), font, 0.75, textColor, 2, LINE_AA)
    putText(newImage, f"PASTA v1 COUNT: {len(goodContours)} PASTAS", (20, 360), font, 0.75, textColor, 2, LINE_AA)
    putText(newImage, f"Pasta v2 COUNT: {len(blobs)} PASTAS", (20, 390), font, 0.75, textColor, 2, LINE_AA)


    # draw crosshairs and box
    line(newImage, (targetX - 20, targetY), (targetX + 20, targetY), lineColor, 2)
    line(newImage, (targetX, targetY - 20), (targetX, targetY + 20), lineColor, 2) 
    rectangle(newImage, (targetX - xMargin, targetY - yMargin), (targetX + xMargin, targetY + yMargin), lineColor, 2)
    rectangle(newImage, (targetX - xBorder, targetY - yBorder), (targetX + xBorder, targetY + yBorder), lineColor, 2)

    #draw macaronis on to image
    for blob in blobs:
        circle(newImage, (int(blob.pt[0] + targetX - xBorder), int(blob.pt[1]) + targetY - yBorder), int(blob.size / 2), lineColor, 5)
    drawContours(newImage[targetY - yBorder:targetY + yBorder, targetX - xBorder:targetX + xBorder], goodContours, -1, lineColor, 3)

    # combine images again
    newImage = hconcat([newImage, processImage])
    putText(newImage, "Contour", (newImage.shape[1] - processImage.shape[1], 20), font, 0.75, textColor, 2, LINE_AA)
    putText(newImage, "Contrast", (newImage.shape[1] - processImage.shape[1], int(newImage.shape[0] / 2) + 20), font, 0.75, textColor, 2, LINE_AA)

    # display
    imshow("PASTA", newImage)

    key = waitKey(1)
    #if a key was pressed
    if key != -1:
        # press x to exit
        if key == ord('x'):
            destroyAllWindows()
            break
        # press p to save an image
        elif key == ord('p'):
            time = datetime.now()
            imwrite(f"{time.year}_{time.month}_{time.day}_{time.hour}_{time.minute}_{time.second}.png", newImage)

        # w/s to change contour sensitivity
        elif key == ord('w'):
            tMin += 5
        elif key == ord('s'):
            tMin -= 5

        # i/o change minArea
        elif key == ord('i'):
            xMargin += 1
            yMargin += 1
        elif key == ord('o'):
            xMargin -= 1
            yMargin -= 1

        # d/a change contour sensitivity
        elif key == ord('d'):
            tMax += 5
        elif key == ord('a'):
            tMax -= 5

        # k/j change scope of detection
        elif key == ord('k'):
            xBorder += 5
            yBorder += 5
        elif key == ord('j'):
            xBorder -= 5
            yBorder -= 5

        # m/n changes contrast
        elif key == ord('m'):
            alpha += 0.1
            alpha += 0.1
        elif key == ord('n'):
            alpha -= 0.1
            alpha -= 0.1
