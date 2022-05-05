import cv2
import numpy as np
import argparse
#import matplotlib.pyplot as plt
import json
import os
#from scipy.spatial import distance
import glob
from skimage.transform import resize
from scipy.spatial import distance
import multiprocessing as mp
from multiprocessing import Pool
import time
import re
import warnings


warnings.filterwarnings("ignore")

start = time.perf_counter()
"""
This function defines the command-line arguments
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help = "Figures input directory")
    parser.add_argument("--amazonDirectory", help = "Amazon label bounding boxes")
    parser.add_argument("--outputDirectory", help = "directory to save the segmented images")
    parser.add_argument("--jsonDirectory", help = "directory to save the json output")
    parser.add_argument("--TransformerDirectory", help = "directory to store the output of the Transformer model")
    parser.add_argument("--processingDirectory", help = "directory to store the output of resizing the images")
    parser.add_argument("--jsonFilename", help = "Name of output json file")
    
    return parser

"""This function gets the amazon directories to ensure proper alignment with figure files, loads 
the json file for the current figure file.
"""    
def get_amazonFiles(index):
    
    parser = get_args()
    args = parser.parse_args()
    amazon_dir = args.amazonDirectory
    amazon_files = os.listdir(amazon_dir)
    # sort the amazon filenames
    #amazon_files.sort()
    current_ama = amazon_files[index]
    #print(current_ama)
    amazon_fpath = os.path.join(amazon_dir, current_ama)
    try:
        # check if the amazon file in not empty
        if os.stat(amazon_fpath).st_size != 0:
            label_path = open(amazon_fpath, 'r', encoding = 'utf-8')
            label_info = json.load(label_path)
        
            return label_info, current_ama
    except Exception as error:
        print(error)

# """This function sorts the figure files to properly align with the amazon files"""
# def get_imageFiles():
#     parser = get_args()
#     args = parser.parse_args()
#     img_dir = args.file_path
#     img_paths = os.listdir(img_dir)
#     # sort the amazon filenames
#     #img_paths.sort() 
#     return img_paths

"""This function takes in the label info and returns label
bounding box and label name
"""

def extract_label_bboxes(index):
    label_info, current_ama = get_amazonFiles(index)
    
    texts = []
    try:
        
        if type(label_info) is dict and "TextDetections" in label_info.keys():
            contents = label_info['TextDetections']
            for i, j in enumerate(contents):
                if j['Type'] == 'LINE' and j["Confidence"] >= 75:
                    bbox = j['Geometry']['BoundingBox']
                    detectedtext = j['DetectedText']
                    each_text = {"detectedtext":detectedtext, "geometry":bbox}
                    texts.append(each_text)
        else:
            for i, j in enumerate(label_info):
                if j['Type'] == 'LINE'and j["Confidence"] >= 75:
                    bbox = j['Geometry']['BoundingBox']
                    detectedtext = j['DetectedText']
                    each_text = {"detectedtext":detectedtext, "geometry":bbox}
                    texts.append(each_text)
    except Exception as error:
        print(error)
    
    return texts, current_ama      

def getLabelBoundingBox(index):
    texts, current_ama = extract_label_bboxes(index)
    part1_copy = {}
    part2_copy = {}
    each_part1 = {}
    each_part2 = {}
    part1_all = []
    part2_all = []

    results = []
    each_result = {}
    try:

        for i in range(0, len(texts)):
            text = texts[i]
            index = i
            
            #***************** find 'FIG. 6' or similar as a whole
            a = re.findall("[F][[A-Za-z]?[G|g][.|,]?[ ]*[0-9][0-9]?[ ]*[0-9]?[0-9]?", text['detectedtext'])  # *:  repeat 0 to many times, +: 1 to many, ? :0 or 1 times, [0-9]? is together, so 1 number or two numbers
            #***************** find 'Figure 3' or similar as a whole
            b = re.findall("Figure"+"[.|,]?"+"[ ]*[0-9][0-9]?", text['detectedtext'])
            #***************** find '1.3' or similar as a whole
            c = re.findall('[0-9][0-9]?[.|,][0-9][0-9]?', text['detectedtext'])
            #***************** find 'FIGURE 3' or similar as a whole
            d = re.findall("FIGURE"+"[.|,]?"+"[ ]*[0-9][0-9]?", text['detectedtext'])
            
            
            #*********case a******** find 'FIG. 6' or similar as a whole
            if a:
                label = "".join(a)
                label = label.replace(" ", "")
                each_result = {"label":label, "geometry":text['geometry']}
                results.append(each_result)
                # print('a:', each_result)
            
                
            #***********case b****** find 'Figure 3' or similar as a whole
            elif b:
                label = "".join(b)
                label = label.replace(" ", "")
                each_result = {"label":label, "geometry":text['geometry']}
                results.append(each_result)
                # print('b:', each_result)
            
            elif d:
                label = "".join(d)
                label = label.replace(" ", "")
                each_result = {"label":label, "geometry":text['geometry']}
                results.append(each_result)
            #***********case c****** find '1.3' or similar as a whole
            elif c:
                label = "".join(c)
                each_result = {"label":label, "geometry":text['geometry']}
                results.append(each_result)
                # print('c:', each_result)
        
            #***********case d****** when 'FIG. 4' is splitted, for example, ['4', 'FIG.']
            elif len(texts)==2:
                part1 = re.findall("[F][[A-Za-z]?[G|g][.|,]?[ ]*", text['detectedtext'])
                part2 = re.findall("[0-9][0-9]?", text['detectedtext'])   # 20180319  =>> ['20', '18', '03', '19']
            
                # print('part1:', part1)
                # print('part2:', part2)
                
                part1 = "".join(part1)
                part2 = "".join(part2)
                
                geometry = text['geometry']
                
                if part1 != '':
                    
                    part1_copy = {"part":part1, "geometry":geometry}
                    
                if part2 != '':
                    part2_copy = {"part":part2, "geometry":geometry}
                    
                    
            #***********case e****** when there are more elements in the extracted text, but 'FIG. x' is splitted and thus can't be solved by a. 
            # for example,  ['fn', '1_1/_I', '7', '6', 'FIG.', 'FIG.']
            # for example,  ['00000', 'NFC', '00000', 'FIG. 3']    # this is solved by both case a and case e
            # for example,  ['10', 'C:sering...', '5:51', 'FIG.', "20180319 Estmdted Tas'r R: competion ry a", 'Segencing...'] 
            elif len(texts) > 2:
                part1 = re.findall("[F][[A-Za-z]?[G|g][.|,]?[ ]*", text['detectedtext'])
                part2 = re.findall("[0-9][0-9]?", text['detectedtext'])
            
                # print('part1:', part1)
                # print('part2:', part2)
                
                part1 = "".join(part1)
                
                if len(part2) == 1:
                    part2 = "".join(part2)
                    # print('22222')
                else:
                    part2 = ''
                    
                # print('part2:', part2)
                
                if part1 != '':
                    each_part1 = {"label":part1, "geometry":text['geometry']}
                    part1_all.append(each_part1)
                    #print('part1 in e :', part1_all)
                    
                if part2 != '':
                    each_part2 = {"label":part2, "geometry":text['geometry']}
                    part2_all.append(each_part2)
                    #print('part2 in e :', part2_all)


        # put parts together
        if part1_copy != {} and part2_copy != {}:      
            complete_label = part1_copy['part'] + part2_copy['part']   
            
            # print('complete_label:', complete_label)
            each_result = {"label":complete_label, "geometry":part1_copy['geometry']}
            results.append(each_result)
            

        # put parts together
        if len(part1_all) <= len(part2_all):
            for i in range(0, len(part1_all)):
                if part1_all[i]["label"] != '' and part2_all[i]["label"] != '':      
                    complete_label = part1_all[i]["label"] + part2_all[i]["label"]  
                
                    each_result = {"label":complete_label, "geometry":part1_all[i]['geometry']}
                    results.append(each_result)

        elif len(part1_all) > len(part2_all):
            for i in range(0, len(part2_all)):
                if part1_all[i]["label"] != '' and part2_all[i]["label"] != '':      
                    complete_label = part1_all[i]["label"] + part2_all[i]["label"]   
                
                    each_result = {"label":complete_label, "geometry":part1_all[i]['geometry']}
                    results.append(each_result)
    except Exception as error:
        print(error)
    
    return results, current_ama

    
"""
This function takes in the index of the figure label, extracts the dimensions
of the label coordinates, and converts it back to the original unit.
We import extract_label_bboxes from Amazon_label
"""
def label_points(index):
    parser = get_args()
    args = parser.parse_args()
    img_dir = args.file_path
    #img_paths =  os.listdir(img_dir)   # list of figure files
    label_and_bbox, current_ama = getLabelBoundingBox(index)
    
    label_conv_points = []
    label_names = []
    # get the current figure file
    #current_img = img_paths[index]
    current_img = current_ama[:-4] + "png"
    #img_path = os.path.join(img_dir, current_img)
    
    try:
        img = cv2.imread(os.path.join(img_dir, current_img))
    
        for info in label_and_bbox:
            if img is not None:
                h, w = img.shape[:2]
                # get the label name
                label_name = info['label']
                # get the bounding box
                geometry = info['geometry']
                width, height = geometry['Width'], geometry['Height']
                left, top = geometry['Left'], geometry['Top']
    
                # convert back the amazon coordinates to the original coordinates
                width = int(width * w)
                height = int(height * h)
                left = int(left * w)
                top = int(top * h)
                label_conv_points.append((left, width, top, height))
                label_names.append(label_name)
            else:
                break    
    except Exception as error:
        print(error)
    return label_conv_points, label_names, current_img


"""The function below helps to calculate the distance of the labels 
which will be used to find the distance between the images and the labels
"""
def calc_label_center(index):
    label_cent = []
    label_coord, label_names, _= label_points(index)
    try:
        
        # case for a single image
        if len(label_coord) == 1:
            left, width, top, height = label_coord[0]
            # Obtain the center coordinates by adding the top to the height / 2, and adding left to width / 2
            ptX, ptY = (width / 2 + left), (height / 2 + top)
            # save it in the label_cent list
            label_cent.append((ptX, ptY))
        
        else:   
            # case for image with subfigures
            for coord in label_coord:
                left, width, top, height = coord
                # Obtain the center coordinates by adding the top to the height / 2, and adding left to width / 2
   
                ptX, ptY = (width / 2 + left), (height / 2 + top)
                # save it in the label_cent list
                label_cent.append((ptX, ptY))
    except Exception as error:
        print(error)
    
    return label_cent, label_names

"""This function calculates the distance between the image and the labels"""
# # find the distance between the label coordinates and the image coordinates
def AmazonDist_label_image(image_mid, label_cent):
    D = {}
   
    try:
        
        if len(label_cent) == 1:
            # calculate the distance between the label and image
            dist = round(distance.euclidean(label_cent[0], image_mid), 2)
            D[0] = dist
        else:  
            # distance for image with subfigures   
            # loop through the label coordinates and unpack the coordinates
            for ind1, lab in enumerate(label_cent):
            
                # calculate the distance between the label and image
                dist = round(distance.euclidean(lab, image_mid), 2)
                D[ind1] = dist      
    except Exception as error:
        print(error)
    return D


""" 
This function loads the image and wipe out the labels 
using label coordinates from Amazon Rekognition tool
"""
def figure_only(index):
        
    # get patent id
    parser = get_args()
    args = parser.parse_args()
    #output_dir = args.outputDirectory  # directory to save the segmented figures
    label_coord, lb_names, img_path = label_points(index)
    img_dir = args.file_path   # figures directories  
    image = cv2.imread(os.path.join(img_dir, img_path))
    try:    
       # case for a single image
       if len(label_coord) == 1:
           # unpack the coordinates
           left, width, top, height = label_coord[0]
           # set the pixels in those location to white to wipe out the labels
           image[(top - 1) : (top + height + 1), (left - 1): (left + width + 1)] = (255, 255, 255)
        
          # cv2.imwrite(os.path.join(output_dir, img_path), image)
       else:  
            
           # case for image with subfigures
           for label in label_coord:
               # unpack the coordinates
               left, width, top, height = label
               # set the pixels in those location to white to wipe out the labels
               image[(top - 1) : (top + height + 1), (left - 1): (left + width + 1)] = (255, 255, 255)
            
          # cv2.imwrite(os.path.join(output_dir, img_path), image)
    except Exception as error:
        print(error)

    return image, img_path, lb_names


"""This function takes the figure without labels and resize it so we can apply the
transformer model on it."""
def preprocessing(index):
    parser = get_args()
    args = parser.parse_args()
    process_dir = args.processingDirectory
    try:

        img, img_path, _ = figure_only(index)
        image = img.copy()
    
        #### Saving input images #################
        original_res= resize(image, (128, 128, 3), mode='constant', preserve_range=True)
        cv2.imwrite(os.path.join(process_dir, img_path), original_res)

        #return original_res, img_path
    except Exception as error:
        print(error)
    

parser = get_args()
args = parser.parse_args()
img_paths = args.file_path
rel_paths = os.listdir(img_paths)

#This runs the processing in parallel across the cpu cores
# if __name__ == "__main__":
#     indices = list(range(len(rel_paths)))
#     p = mp.cpu_count()   # count the number of cpus
#     process = Pool(p)
#     result = process.map(preprocessing, indices)
#    # result = process.map(figure_only, indices)
#     process.close()
#     process.join()

# finish = time.perf_counter()

# print("Finished in {} seconds".format(round(finish - start), 2))
