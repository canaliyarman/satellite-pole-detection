#  INSTALLATION
I recommend using my environment due to CUDA and Pytorch versioning clashes in the original documentation. However original installation steps can be found here https://mmrotate.readthedocs.io/en/latest/install.html#installation

To recreate the conda environment
`conda env create -f mmrotate.yaml` or 
Clone the original repository and build it

    git clone https://github.com/open-mmlab/mmrotate.git
    cd mmrotate
    pip install -v -e .


Implementations of all models are located in custom directory. Change the path in `/_base_/datasets/custom_detection.py` to the Transformed ETDII Dataset path all necessary places. Change the path in dataset paths and model weights in all python files in the `custom` directory
Move the `/custom/`and `/base/` directories to `mmrotate/configs` directory.

If you encounter any issues with the installation please contact me, there are some issues with CUDA compatibility, CUDA11.8 is recommended with torch 2.4.0

# Training and Testing
To train the models run, change the config file to your desired model

    python tools/train.py configs/custom/custom_roi_transformer_lsknet.py
To test the models run, change the config file to your desired model

    python tools/test.py configs/custom/custom_swin.py ./work_dirs/custom_swin/latest.pth --show-dir work_dirs/custom_swin_vis --eval mAP

# Inference
Inference notebook is used for inferring on a singular image. Make sure to adjust paths.
