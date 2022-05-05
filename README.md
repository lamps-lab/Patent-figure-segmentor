# PatentFigureSegmentation
This project is about the segmentation of patent drawings using a pipeline integrated with Transformer model. 
The following steps were carried out in order to perform the segmentation:

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
4. From the root directory, run the command below:
      - python3 processing_updated.py <image_path> --amazonDirectory <amazon_filepath> --processingDirectory </processing_dir/ends/with/img/created/in/step3>

5. Next step is to run the transformer on the processed images. run the command below:
    - python3 test_ex.py --loaddirec "MedT.pth" --val_dataset "processing_dir/excluding/img/directory/created/in/step3" --direc 'where/to/save/transformer/result' --batch_size 1 --modelname "MedT" --imgsize 128 --gray "no"

6. Finally, to segment the images, run the command below:
    - python3 segmentImage_json.py <image_path> --amazonDirectory <amazon_filepath> --TransformerDirectory <path/where/you/saved/transformer/result> --jsonDirectory <path/to/save/json/file> --outputDirectory <path/to/save/segmented/images>

## Example
- We have included few images and their corresponding amazon bounding box information in the **test** folder to test the pipeline.  

# Docker Build
- A Dockerfile is provided with the Python 3.8 library. This will create a working directory called **patent** with all the 
project dependencies installed.

- Build container: docker build -t <name-of-image> . e.g. kehindeajayi01/patentfigure:patent, the dot after your docker image implies your current directory.

- Run the container interactively, mount this project dir to /patent/: docker run -it --name <patent> patentfigure:patent

