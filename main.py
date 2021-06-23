import cv2
import pytesseract
import numpy as np


def empty(a):
    pass


# Tesseract is used to read text in images
# Must be instaled in the computer


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# A screen in order to change on air some parameters
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 73, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 34, 255, empty)


# Get Contours and detect squares
def getContours(img, imgContour):
    global imgCropped
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Find contours

    # imgDrawCount = imgContour.copy()  # Just an image to draw (Disabled)
    for cnt in contours:

        area = cv2.contourArea(cnt)  # Calculate area for every contour
        peri = 0.02 * cv2.arcLength(cnt, True)  # Calculate perimeter for every contour
        approx = cv2.approxPolyDP(cnt, peri, True)

        if 100000 > area > 45000 and len(approx) == 4:  # Search for rectangles with a concrete area
            # Color detection would be nice to have

            x, y, w, h = cv2.boundingRect(approx)

            # Code to draw a bounding rectangle and its area
            # Its disabled because its affects tesseract reads

            # cv2.rectangle(imgDrawCount, (x, y), (x + w, y + h), (255, 0, 255), 7)
            # cv2.putText(imgDrawCount, str(area), (x, y - 5), 1, 1.5, (0, 255, 0), 2)

            # Code to draw a tilted bounding rectangle.
            # Its disabled because its affects tesseract reads

            # rect = cv2.minAreaRect(cnt)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(imgContour, [box], 0, (0, 0, 255), 2)

            imgCropped = imgContour[y:y + h, x:x + w]  # Crop image using bounding rect coordinates

    return imgCropped


while True:

    path = "Resources/Paquete1.jpg"  # Path of the img
    img = cv2.pyrDown(cv2.imread(path))  # Read img and scale it down

    # Next lines are just for preprocessing the image

    imgContour = img.copy()
    cropped = img.copy()

    imgBlur = cv2.GaussianBlur(img, (5, 5), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    # Get values from sliders
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    # Continue with preprocessing
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    # Call getContours function and store it in 'imgCroped'
    imgCropped = getContours(imgDil, imgContour)

    # Just some stuff to scale down the image
    scale_percent = 50  # percent of original size
    width = int(imgContour.shape[1] * scale_percent / 100)
    height = int(imgContour.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(imgContour, dim, interpolation=cv2.INTER_AREA)

    # Show images
    cv2.imshow("Result", resized)
    cv2.imshow("Result2", imgCropped)

    # Read text form cropped image
    imgRead = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2RGB)  # tesseract works with RGB
    string = pytesseract.image_to_string(imgRead)  # Read and put text in a string
    with open('Data.txt', 'w') as f:  # Create/ open 'Data.txt'
        f.write(string)  # Wirte string in txt
        f.write('\n')

    print(string)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
