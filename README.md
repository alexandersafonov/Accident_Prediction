# Anticipating Car Accidents using physical features
For videos and original method, refer to:
https://aliensunmin.github.io/project/dashcam/

For using QRNN instead of LSTM, refer to the following link:
https://github.com/salesforce/pytorch-qrnn

For Unsupervised Depth Estimation, refer to:
https://github.com/google-research/google-research/tree/master/depth_from_video_in_the_wild


# Organization:
- Agent locations are obtained in World Coordinates in notebooks contained in the "World_Coordinate_Extraction" folder
- The original features and world coordinates are saved in the "Features" folder
- Pretrained models are saved in the "model" folder
- Accident Prediction Methods are contained in the "Accident_Prediction_Notebooks" folder

the 'torch_accident_wc_appearance.ipynb' notebook can be run to train or evaluate the latest method
the original method is replicated in torch in the torch_accident_DSA.py file











