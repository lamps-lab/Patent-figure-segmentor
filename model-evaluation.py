import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_images", help = "path to the test images")
    parser.add_argument("--transformer_dir", help = "path to the transformer results")
    parser.add_argument("--annotation", help = "path to the annotation json file")
    parser.add_argument("--output_dir", help = "path to save the segmented images or images with contour")
    args = parser.parse_args()
    return args

def get_boundingbox():
    print('Preparing to get image bounding boxes')
    args = get_args()
    # Test images
    img_dir = args.test_images
    image_names = os.listdir(img_dir)
    output_dir = args.output_dir
    
    # Transformer images
    transformer_dir = args.transformer_dir
    all_pred_coordinates = {}
    predicted_coordinates = []
    names = []
    
    try:
        # Read the original and prediction images
        for i, img_name in enumerate(image_names):
            img = cv2.imread(os.path.join(img_dir, img_name))
            
            # append the image name
            names.append(img_name)
            # read transformer image
            transformer_image = cv2.imread(os.path.join(transformer_dir, img_name))
        
            """
            get the height and width of the original image: this will be used for converting the resized image 
            back to the original dimension 
            """
            orx = img.shape[1]
            ory = img.shape[0]
            scalex = orx / 128
            scaley = ory / 128
        
            # Added code by me
            gray = cv2.cvtColor(transformer_image, cv2.COLOR_BGR2GRAY)
            canny_get_edge = cv2.Canny(gray, 40, 250)
            # Perform a little bit of morphology:
            # Set kernel (structuring element) size:
            kernelSize = (3, 3)
            # Set operation iterations:
            opIterations = 1
            # Get the structuring element:
            morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
            # Perform Dilate:
            morphology = cv2.morphologyEx(canny_get_edge, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101) # preds
            contours, hierarchy = cv2.findContours(morphology, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
            
            for c in contours:
                rect = cv2.boundingRect(c)
                if rect[2] < 50 or rect[3] < 50: continue
                cv2.contourArea(c)
                x, y, w, h = rect
                x = int(x*scalex)
                y = int(y*scaley)
                w = int(w* scalex)
                h = int(h * scaley)
                image_bounding_box = (x, y, w, h)
                predicted_coordinates.append(image_bounding_box)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #ROI = img[y:y + h, x:x + w]
        
                cv2.imwrite(os.path.join(output_dir, img_name), img)
            
            all_pred_coordinates[img_name] = predicted_coordinates
        return all_pred_coordinates, names
              
    except Exception as error:
        print(error)
    print('**************************** Resized the bounding Box ******************************* ')


def get_annotation():
    coordinates={}
    coordinate=[]
    all_pred_coordinates, names = get_boundingbox()
    # read annotation in json file
    args = get_args()
    annotation_path = args.annotation
    annotation_df = pd.read_json(annotation_path)

    annotation_transpose = annotation_df.transpose().reset_index()[['filename', 'regions']]
    try:
        
        for img in names:
            element = annotation_transpose[annotation_transpose['filename'] == img]
            a = element.index
            a = a[0]
            size = len(element['regions'][a])
            for j in range(size):
                current = element['regions'][a][j]['shape_attributes']
                annotation_bbox = (current['x'], current['y'], current['width'], current['height'])
                coordinate.append(annotation_bbox)
            coordinates[img]=coordinate
        return coordinates, all_pred_coordinates, names
    except Exception as error:
        print(error)

def IOU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0 or h_intersection <= 0:  # No overlap
        return 0
    I = w_intersection * h_intersection
    U = w1 * h1 + w2 * h2 - I  # Union = Total Area - I
    return I / U


def iou_pred(y_test, y_pred):
    z = len(y_test)
    z1 = len(y_pred)
    result = []
    for i in range(len(y_test)):
        for j in range(len(y_pred)):
            result.append(IOU(y_test[i], y_pred[j]))
    results = sorted(result, reverse=True)
    if len(y_test) == len(y_pred):
        fresult = results[:z]
    else:
        fresult = results[:z1]
    p1 = [1 if i > .7 else 0 for i in fresult]
    # if z != z1:
    #     return 0
    if sum(p1) / len(p1) == 1:
        return 1
    else:
        return 0

if __name__ == '__main__':
    coordinates, all_pred_coordinates, names = get_annotation()
    total_result = 0
    for img_name in names:
        total_result += iou_pred(coordinates[img_name], all_pred_coordinates[img_name])

    print('Total accuracy', total_result / len(names))