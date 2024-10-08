import math
import cv2
import numpy as np

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    return imgGrayscale, imgThresh

def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    return imgValue

def maximizeContrast(imgGrayscale):
    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations=10)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations=10)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat

def rotation_angle(linesP):
    angles = []
    lengths = []
    for i in range(0, len(linesP)):
        l = linesP[i][0].astype(int)
        p1 = (l[0], l[1])
        p2 = (l[2], l[3])
        doi = (l[1] - l[3])
        ke = abs(l[0] - l[2])
        angle = math.atan(doi / ke) * (180.0 / math.pi)
        if abs(angle) > 45:
            angle = (90 - abs(angle)) * angle / abs(angle)
        angles.append(angle)
        lengths.append(math.sqrt((l[0] - l[2])**2 + (l[1] - l[3])**2))
        print(f"Line {i}: p1={p1}, p2={p2}, angle={angle}, length={lengths[-1]}")

    if not angles:
        return 0.0
    weighted_avg_angle = np.average(angles, weights=lengths)
    print(f"Final rotation angle: {weighted_avg_angle}")
    return weighted_avg_angle

def rotate_LP(img, angle):
    height, width = img.shape[:2]
    ptPlateCenter = width / 2, height / 2
    rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
    rotated_img = cv2.warpAffine(img, rotationMatrix, (width, height))
    return rotated_img

def Hough_transform(threshold_image, nol=6):
    h, w = threshold_image.shape[:2]
    linesP = cv2.HoughLinesP(threshold_image, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is None:
        return None
    dist = []
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        d = math.sqrt((l[0] - l[2])**2 + (l[1] - l[3])**2)
        if d < 0.5 * max(h, w):
            d = 0
        dist.append(d)

    dist = np.array(dist).reshape(-1, 1, 1)
    linesP = np.concatenate([linesP, dist], axis=2)
    linesP = sorted(linesP, key=lambda x: x[0][-1], reverse=True)[:nol]
    return linesP

def main():
    Min_char = 0.01
    Max_char = 0.09
    LP_img = cv2.imread('doc/cropped_LP2.png')
    _, thresh = preprocess(LP_img)
    linesP = Hough_transform(thresh)
    if linesP is not None:
        for l in linesP:
            print(l)
    cv2.imshow('thresh', thresh)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
