# POLE DETECTION DISSERTATION CODE
I've included dataset creation, dataset transformation and different versions of object detection implementations and training code in this repo. There are detailed steps inside the directories on how to use them. 

![DetectionResults]("Detection Results.png")

## etdii-dataset-transformer
Intended use is on google colab as it utilizes google drive to store the images however can be used in any environment with a few adjustments. Transforms original ETDII dataset to smaller images with oriented bounding box annotations. 
## googlemaps-dataset-creation
Notebook has explanations on how the entire thing works, it captures images of the locations using the locations dataset. 
## yolo-object-detection
There's a README inside both google-maps-detection and etdii-detection that explains how the code works

## mmrotate-object-detection
There's README inside both detection-using-lsknet and mmrotate that explains how to install and run the code, implementations of custom models are inside /custom/ directory. I've had a lot of issues installing the required libraries on my own due to versioning clashes. I included the .yaml files of both conda environments that I've used. You can easily recreate the conda environments and follow a few steps to run them. 

If you encounter any issues please let me know

