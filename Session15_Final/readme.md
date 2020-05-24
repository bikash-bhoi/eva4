## Problem Description:
-  Assignment description :
  -  Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object. 

## Creating Dataset:
-  Created the dataset of ~400k images by overlaying multiple foreground on top of multiple background images. The dataset creation steps and stats [here](https://github.com/bikash-bhoi/eva4/tree/master/Session15).

### Dataset Statistics

| Data | File/Folder Name | Type |Count | Image_size | Datafolder_size | Mean | std |
|---|---|---|---|---|---|---|---|
| Background | bg | Jpg | 102 | 224x224X3 | 1.02 MB |  |   |
| Foreground | fg | png | 104 | Varying size from 90x90X4 to 160x160X4 | 2.25 MB |  |   |
| Foreground Mask | Cars_Mask | png | 104 | Varying size from 90x90X4 to 160x160X4 | 102 KB |  |   |
| Foreground over Background | fg_bg.zip | Jpg | 424320 | 224x224X3 | 7 GB | [0.3931, 0.3785, 0.3606] | [0.1965, 0.1813, 0.1779] |
| Foreground over Background Mask | fg_bg_mask.zip | Jpg | 424320 | 224x224X1 | 2 GB | [0.1630] | [0.3598] |
| Depth Output | depth_output.zip | Jpg | 424320 | 224x224X1 | 764 MB | [0.0878] | [0.0157] |

## Handling Large Dataset:
- Tried LMDB format to store and load images. Even though Training Seemed faster, It was using more space for colab to handle so went ahead with normal zip-unzip trick. 
- The data was zipped in google drive, While loading colab, the images folder were being unzipped to Colab local file system for Train/ Test.
- Logs and the models were directly saved to colab to overcome loss of data due to runtime disconnection.

## Creating Train test Split:

- The Images were in theier respective folders, Train- Test Images were recognized by a one time generated csv files.(train.csv/ test.csv) [Code here](https://github.com/bikash-bhoi/eva4/blob/master/Session15_Final/EVA4_Session15_Train_test_split.ipynb)

## Models Chosen :
- Depth Prediction: Resnet50 with strided convolution for up-projection, Number of parameters: 
<br>Total params: 67,569,473
<br>Trainable params: 67,569,473
<br>Non-trainable params: 0

- Mask Prediction : Auto Encoder Decoder, Number of parameters : 
<br>Total params: 16,393,752
<br>Trainable params: 16,393,752
<br>Non-trainable params: 0

## Output:
- Training Notebook : [Here](https://github.com/bikash-bhoi/eva4/blob/master/Session15_Final/EVA4_Session15_2.ipynb)
- The outputs obtained are not so great . below are the outputs at the end of # of epochs with RMSE.
- Models were trained with 32x32 images for first 10 epochs and with 64x64 with later 10 epochs. 64x64 demanded a lesser Batch size increasing the training time.
- Loss function used : CosineSimilarity
- Accuracy calculation : RMSE of pixel values between predicted Depth and ground Truth. Pixel Values range : [0,1]
- 

### Epoch 5 (RMSE : 0.18184852525797063)
![mask](https://github.com/bikash-bhoi/eva4/blob/master/Session15_Final/output_images/mask_0_4.jpg) ![depth](https://github.com/bikash-bhoi/eva4/blob/master/Session15_Final/output_images/depth_0_4.jpg)
### Epoch 10 (RMSE : 0.17910172811893638)
![mask](https://github.com/bikash-bhoi/eva4/blob/master/Session15_Final/output_images/mask_5_9.jpg) ![depth](https://github.com/bikash-bhoi/eva4/blob/master/Session15_Final/output_images/depth_5_9.jpg)
### Epoch 15 (RMSE : 0.05212091282534608)
![mask](https://github.com/bikash-bhoi/eva4/blob/master/Session15_Final/output_images/mask_10_14.jpg) ![depth](https://github.com/bikash-bhoi/eva4/blob/master/Session15_Final/output_images/depth_10_14.jpg)
### Epoch 20 (RMSE : 0.04344806143256287)
![mask](https://github.com/bikash-bhoi/eva4/blob/master/Session15_Final/output_images/mask_15_19.jpg) ![depth](https://github.com/bikash-bhoi/eva4/blob/master/Session15_Final/output_images/depth_15_19.jpg)
### Link to train log [here](https://drive.google.com/file/d/1P7L8cEYY1BFPG7gx22mF_6VsYT2sM_qn/view?usp=sharing)


## Issues faced

