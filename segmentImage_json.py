import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import os
import time
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from PIL import Image,ImageDraw
import time
#from skimage.io import imread_collection
import argparse
import glob
from processing_updated import get_args, figure_only, extract_label_bboxes
#from output_new import finetune_label
from processing_updated import calc_label_center
from processing_updated import AmazonDist_label_image
import re 
from multiprocessing import Pool, Process
import multiprocessing as mp
import time
import json


start = time.perf_counter()

def resize_boundingbox(index):
    #print('Image is going to processed : ', index)
    parser = get_args()
    args = parser.parse_args()
    # original images
    #img_dir = args.file_path
    img_only, img_path, label_name = figure_only(index)
    
    output_dir = args.outputDirectory
    # Transformer prediction images
    transformer_dir = args.TransformerDirectory
    transf_img_rel_paths = os.listdir(transformer_dir)
    transf_pred_path = os.path.join(transformer_dir, img_path)
    #all_coordinates = []
    filename = os.path.join(output_dir, img_path)    
    # Read the original and prediction images
    json_name = {}
    sub_list = []

    try: 
        
 #       img_orig = cv2.imread(img_path)
        img = img_only.copy()
        pred_orig = cv2.imread(transf_pred_path)
        preds = pred_orig.copy()
    
        """
        get the height and width of the original image: this will be used for converting the resized image 
        back to the original dimension 
        """
        orx = img.shape[1]
        ory = img.shape[0]
        scalex = orx / 128
        scaley = ory / 128
       
        # Added code by me
        gray = cv2.cvtColor(preds, cv2.COLOR_BGR2GRAY)
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
        im = img_only.copy()  
        
        json_name['patent_id'] = os.path.splitext(img_path)[0]   # [:19]
        json_name["Figure_file"] = img_path
        json_name["n_subfigures"] = len(label_name)
        
        #minDistIndex = -float('inf')
        default = "999" 
        defaultNum = 1   
        for c in contours:
            D = {}
            sub_file = {}
            rect = cv2.boundingRect(c)
            if rect[2] < 5 or rect[3] < 5: continue
            cv2.contourArea(c)
            x, y, w, h = rect
            x = int(x*scalex)
            y = int(y*scaley)
            w = int(w* scalex)
            h = int(h * scaley)
           
            (ptX, ptY) = (w / 2 + x), (h / 2 + y)  # center of the figure
            image_mid = (ptX, ptY)
            label_cent, _ = calc_label_center(index)   # center of the labels
            D = AmazonDist_label_image(image_mid, label_cent)   # calculating the distance between the figure and labels
            minDistIndex = min(D, key=D.get, default = "empty")            # index of label with the smallest distance to the figure
              
            if minDistIndex == 'empty':              # if label is not matched, we put default number
            
                labelNum = int(default + str(defaultNum))
                sub_file["subfigure_id"] = labelNum
                sub_file["subfigure_file"] = img_path[:-4] + "_" + str(labelNum) + '.png'
                sub_file["subfigure_label"] = labelNum
                sub_list.append(sub_file)

                ROI = im[y:y + h, x:x + w]
                cv2.imwrite(filename[:-4] + "_{}.png".format(labelNum), ROI)
                defaultNum += 1
            else:
                label = label_name[minDistIndex]             # closest label to the figure
                figureDigit = re.findall('\d*\d+', label) 
                figureDigit = figureDigit[0]
                sub_file["subfigure_id"] = int(figureDigit)
                sub_file["subfigure_file"] = img_path[:-4] + "_" + figureDigit + '.png'
                sub_file["subfigure_label"] = label
                sub_list.append(sub_file)
                # get the subfigure
                ROI = im[y:y + h, x:x + w]
                cv2.imwrite(filename[:-4] + "_{}.png".format(figureDigit), ROI)
        
        json_name['subfigurefile'] = sub_list
        # check = json_name.pop('subfigurefile', 'Not_found')
        # if check == 'Not_found':
        #     json_name.clear()
    except Exception as error:
        #errorList = open("/data/kajay/Errors/errorList_" + jsonFilename + ".txt", 'a')
        #errorList.write(img_path + "\n")
        print(error)
#        print(filename)    
    return json_name
#    print('**************************** Resized the bounding Box ******************************* ')


parser = get_args()
args = parser.parse_args()
json_output = args.jsonDirectory
amazon_paths = args.amazonDirectory
jsonFilename = args.jsonFilename
rel_paths = os.listdir(amazon_paths)
indices = list(range(len(rel_paths)))
#indices = list(range(50))  # just for testing
if __name__ == "__main__":
    fp = open(os.path.join(json_output, jsonFilename + '.json'), 'w', encoding='utf-8')
    p = mp.cpu_count()
    process = Pool(p)
    sample = process.map(resize_boundingbox, indices)
    json.dump(sample, fp, ensure_ascii=False,)
    fp.write('\n')
    process.close()
    process.join()


finish = time.perf_counter()
# print("Finished in {} seconds".format(finish-start))
