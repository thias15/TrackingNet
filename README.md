# Tracking Net

This is TrackingNet, a large-scale dataset for tracking containing more than 30k videos.
If you use this dataset for your work, please cite our corresponding ECCV'18 paper. 

## Download Dataset
### Recommended 


### Experimental
This code has only tested under Windows. You may encounter the following two issues. 
1. Due to slight differences in implementation of OpenCV/FFMPEG we cannot gurantee that the frames will be exactly the same as the ones we provide above which were extracted using Windows.
2. From time to time videos are removed from Youtube. It is possible that you won't be able to obtain the complete dataset.

If above code did not work for you, you are really desperate to get the data and understand the risks, please proceed. 

#### Create Conda Environment
`conda env create -f environment.yml`

#### Download Dataset
`download_train.bat` or `download_train.sh`  
`download_train.bat` or `download_train.sh`