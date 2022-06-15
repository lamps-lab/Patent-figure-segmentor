# DeepPatent2 Figure Segmentation
**Abstract**: Recent advances in computer vision (CV) and natural language processing (NLP) have been driven by exploiting big data to explore many practical applications. However, these research fields are still limited not only by the sheer volume, but also by the versatility and diversity, of the available datasets. CV tasks, such as image captioning, have primarily been carried out on natural images. Despite the efficiency of the state-of-the-art (SOTA) captioning models on natural images, they still struggle to produce accurate and meaningful captions on sketched images. In this paper, we introduce DeepPatent2, a new large-scale dataset for patent figure understanding. This dataset provides more than 2.8 million sketched images with 132,890 object categories extracted from US design patent documents. We demonstrate the usefulness of DeepPatent2 to the CV and NLP communities with two use cases namely, image segmentation and image captioning. We trained a baseline image captioning model on a subset of our dataset to produce captions that focus on identifying the actual object and different viewpoints of patent figures. Our model achieves METEOR and ROUGE-l scores of 62.7\% and 63.4\% on the test data. For the image segmentation task, we used a transformer-based method to segment the technical drawings in DeepPatent2, and achieved a 99.5\% accuracy on the test data. The experiments showcased the potential usefulness of our dataset to facilitate future research such as image grounding on sketched images and technical drawings. The Deeppatent2 dataset is publicly available at https://bit.ly/3xsyAty under the CC BY-NC license.


## Data Description
- The DeepPatent2 dataset contains over 4 million design patent drawings in total (Original and Segmented) obtained from the United States Patent and Trademark Office (USPTO). It spans from year 2007 to 2020. The dataset contains compound drawings with sub-images up to 10. Our dataset contains two sub-directories and 1 file in each **year** directory:
    - **Original**: It contains the extracted patent drawings from USPTO in PNG format
    - **Segmented**: It contains the segmented drawings obtained from applying our segmentation pipeline on the Original drawings
    - design*.json: It contains the metadata obtained from the segmentation. The fields in the metadata include patentID, Figure_file, subfigure_file, caption, aspect, subfigure_label, e.t.c.

## Components of DeepPatent2 Pipeline
- This project is about the segmentation of patent drawings using a pipeline integrated with Transformer model.
## Training the Transformer model
- We recommend that you create a python3 virtual environment, and run the requirements.txt file
1. Before training the transformer model, we remove the labels of each sub-figure from the image using Amazon Rekognition. Run the command below from "Training" directory:
    - python3 figure_only.py --filepath <path/to/image/files> --amazonDirectory <path/to/amazon/files> --outputDirectory <path/to/save/figures/with/no/labels>

2. Next, we use point shooting method to create a segmentation mask which serves as labels for the transformer model. We do this for both training and validation.
   - Create a training folder and validation folder. Inside each folder, create two separate folders and name them "img" and "labelcol".
   - The "img" folder is where we will put the resized images of size 128 by 128 pixels, while "labelcol" is where we put the segmentation masks.
   - Run the point shooting method below to resize and create masks for both training and validation:

    - python3 point-shooting-sketched-images.py --image_dir <path/to/figure/with/no/labels/created/in/step 1> --img <path/to/save/resized/figures> --mask <path/to/save/segmentation/masks>

3. Next we train the transformer model. Run the command below:
    - python train.py --train_dataset <path/to/train/folder> --val_dataset <path/to/validation/folder> --direc <path/to/save/trained model> --batch_size 4 --epoch 400 --save_freq 10 --modelname "MedT" --learning_rate 0.001 --imgsize 128 --gray "no"


## Segmentation with trained Transformer model
- The following steps were carried out in order to perform the segmentation using the trained model:

1. We used Amazon Rekognition tool to obtain bounding box coordinates for the figure labels.

2. We used the coordinates obtained in step 1 to mask off the figure labels.

3. We then processed the figure only images before applying a Transformer segmentation model on the images.

4. Then, we used Transformer model to segment the patent drawings and their corresponding labels.

5. We then used a distance-based method to match each figure label to its corresponding subfigure in the file.

5. Finally, the output are segmented images and a json file that includes the metadata extracted from each sub-figure in the patent drawings.

# Running the Pipeline
1.  Clone this repository and create a python virtual environment and activate it.
2. run: pip install -r requirements.txt
3. To wipe out the labels and process the images for the transformer, create a directory for processing the images, and inside the directory, create another directory and name it **img**.
4. From the "Segment" directory, run the command below:
      - python3 processing_updated.py <image_path> --amazonDirectory <amazon_filepath> --processingDirectory </processing_dir/ends/with/img/created/in/step3>

5. Next step is to run the transformer on the processed images. run the command below:
    - python3 test_ex.py --loaddirec <path/to/trained/model (MedT.pth)> --val_dataset <processing_dir/excluding/img/directory/created/in/step3> --direc <path/to/save/transformer/result> --batch_size 1 --modelname "MedT" --imgsize 128 --gray "no"

6. Finally, to segment the images, run the command below:
    - python3 segmentImage_json.py <image_path> --amazonDirectory <amazon_filepath> --TransformerDirectory <path/where/you/saved/transformer/result> --jsonDirectory <path/to/save/json/file> --outputDirectory <path/to/save/segmented/images> --jsonFilename <name/of/json/file/without/json/extension>

## Example
- We have included few images and their corresponding amazon bounding box information in the **testing_files** folder to test the pipeline.  


