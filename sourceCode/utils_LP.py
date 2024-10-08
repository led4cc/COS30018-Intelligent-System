import cv2
import numpy as np
import yaml
from Preprocess import preprocess, Hough_transform, rotation_angle, rotate_LP

ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2',
              24: '3', 25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}

def create_yaml():
    data_yaml = dict(
        train='../input/vietnamese-license-plate/train',
        val='../input/vietnamese-license-plate/valid',
        nc=1,
        names=['License Plate']
    )
    with open('data.yaml', 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=True)

def crop_n_rotate_LP(source_img, x1, y1, x2, y2):
    w = int(x2 - x1)
    h = int(y2 - y1)
    ratio = w / h
    print(f"Detected license plate with aspect ratio {ratio}")
    if 1.0 <= ratio <= 4.73:
        cropped_LP = source_img[y1:y1 + h, x1:x1 + w]
        cropped_LP_copy = cropped_LP.copy()
        print(f"Cropped license plate shape: {cropped_LP.shape}")

        imgGrayscaleplate, imgThreshplate = preprocess(cropped_LP)
        print(f"Preprocessed license plate shape: {imgThreshplate.shape}")
        
        canny_image = cv2.Canny(imgThreshplate, 250, 255)
        kernel = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(canny_image, kernel, iterations=2)
        print(f"Dilated image shape: {dilated_image.shape}")

        linesP = Hough_transform(dilated_image, nol=6)
        if linesP is None or len(linesP) == 0:
            print("No lines detected by Hough Transform")
            return None, None, None

        for i in range(0, len(linesP)):
            l = linesP[i][0].astype(int)
            print(f"Line detected: {l}")

        angle = rotation_angle(linesP)
        print(f"Detected rotation angle: {angle}")
        
        rotate_thresh = rotate_LP(imgThreshplate, angle)
        LP_rotated = rotate_LP(cropped_LP, angle)
        print(f"Rotated license plate shape: {LP_rotated.shape}")

    else:
        print(f"Invalid aspect ratio: {ratio}")
        angle, rotate_thresh, LP_rotated = None, None, None

    return angle, rotate_thresh, LP_rotated

def main():
    print('This is a utility module.')

if __name__ == "__main__":
    main()
