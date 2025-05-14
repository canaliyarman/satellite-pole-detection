#  INSTALLATION
In order to use the LSKNet backbone, different mmrotate versions are required. I solved this issue by creating two seperate yaml files. To install MMRotate with LSKNet you can follow the steps here https://github.com/zcablii/LSKNet/tree/main or recreate the conda environment I've used, I recommend using my environment due to CUDA and Pytorch versioning clashes in the original documentation.

To recreate the conda environment
`conda env create -f lsknet.yaml` or 
Clone the original repository and install build it

    git clone https://github.com/zcablii/Large-Selective-Kernel-Network.git
    cd Large-Selective-Kernel-Network
    pip install -v -e .

Implementations of RoI Transformer with LSKNet backbone and Oriented R-CNN with LSKNet backbone are located in custom_configured_models directory. Change the path in `/_base_/datasets/custom_detection.py` to the Transformed ETDII Dataset locations all necessary pleaces. Change the path in dataset paths and model weights in `custom_roi_transformer_lsknet.py` and `lsknet_custom.py`(this is oriented r-cnn).
Move the `/custom/`and `/base/` directories to `Large-Selective-Kernel-Network/configs` directory.

If you encounter any issues with the installation please contact me, there are some issues with CUDA compatibility, CUDA11.8 is recommended with torch 2.4.0

# Training and Testing
To train the models run, change the config file to your desired model

    python tools/train.py configs/custom/custom_roi_transformer_lsknet.py
To test the models run, change the config file to your desired model

    python tools/test.py configs/custom/custom_swin.py ./work_dirs/custom_swin/latest.pth --show-dir work_dirs/custom_swin_vis --eval mAP

