# Hyperspectral-Image-Classification
Hyperspectral Image Classification for satellite imagery using deep learning

Set the required patch-size value (eg. 11, 21, etc) in patch_size.py and run the following notebooks in order:   
1) IndianPines_DataSet_Preparation_Without_Augmentation.ipynb   
2) CNN_feed.ipynb (specify the number of fragments in the training and test data in the variables TRAIN_FILES and TEST_FILES)   
3) Decoder_Spatial_CNN.ipynb (set the required checkpoint to be used for decoding in the model_name variable)
