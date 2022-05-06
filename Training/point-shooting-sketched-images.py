import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.transform import resize


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help = "path to the figure_only images")
    parser.add_argument("--img", help = "path to save resized figures without labels")
    parser.add_argument("--mask", help = "path to save masked images")
    args = parser.parse_args()

    return args

def preprocessing():
    print('************************* Preprocessing start and Mask Generation using Point-shooting Method ***************************')
    args = get_args()
    data = args.image_dir
    resized_img = args.img
    labelcol = args.mask
    data_list = os.listdir(data)
    print('\n')
    print('Total Number of Images : ', len(data_list))
    for i in range(len(data_list)):
        print('Start Preprocessing Image No :', i+1) 
        # Load image, grayscale, blur, Otsu's threshold
        img_name = data_list[i]
        img = cv2.imread(os.path.join(data, img_name))  ### reading image
        image=img.copy()
        #### Saving original images #####
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  ### gray conversion
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  ## making 0 or 255 pixel
        ### Black pixel co-ordinate detection
        ii1 = np.nonzero(thresh == 255)
        x = list(ii1[1])
        y = list(ii1[0])
        # Now, loop through coord arrays, and create a circle at each x,y pair
        for xx, yy in zip(x, y):
            cv2.circle(img, (xx, yy), 10, (0, 20, 200), 10)
        #### Saving input images #################
        
        orginal= resize(image, (128, 128, 3), mode='constant', preserve_range=True)
        cv2.imwrite(os.path.join(resized_img, img_name), orginal)  ### Saving resized original Image
        img = resize(img, (128, 128), mode='constant', preserve_range=True) ## resize mask
        img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  ### gray conversion
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        ### Saving the mask images ##############
        cv2.imwrite(os.path.join(labelcol, img_name), mask)  ### Saving Masked Image
    print('*********************************** Preprocessing done *************************')


if __name__ == '__main__':
    preprocessing()
