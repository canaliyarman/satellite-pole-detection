# Dataset
The Transformed ETDII dataset needs to be converted to YOLO OBB and YOLO formats in order to train the models, both YOLO OBB and YOLO are supported.
Download or locate the original Transformed ETDII dataset, copy it to this directory and run the convert_yolo_obb.py script to create obb format. 
To create the horizontal annotation dataset run convert_yolo.py after you created the YOLO obb dataset.

# Training 
Change the path in the etdii.yaml for the dataset you wish to use, don't forget to change the path for root for the corresponding model you're using (obb or horizontal).
Run train_yolo_obb.py script to train the obb model or you can use the notebook.
Run train_yolo.py script to train the hbb model or you can use the notebook.
Make sure to change the weight and directory paths, runs directory contains my previous runs and tests. I've also provided the default ultralytics weights.

# Test and visualization
Rest of the python scripts and notebooks are for visualization purposes.
