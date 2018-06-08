# Tiny Face Detector class in TensorFlow
 This code is a trial to revamp the code [Tiny_Faces_in_Tensorflow](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow) which is a TensorFlow port(inference only) of Tiny Face Detector from [authors' MatConvNet codes](https://github.com/peiyunh/tiny).
 
 + dlib face aligner is added


## Converting a pretrained model

`matconvnet_hr101_to_pickle` reads weights of the MatConvNet pretrained model and
write back to a pickle file which is used in a TensorFlow model as initial weights.

1. Download a [ResNet101-based pretrained model(hr_res101.mat)](https://www.cs.cmu.edu/%7Epeiyunh/tiny/hr_res101.mat)
from the authors' repo.

2. Convert the model to a pickle file by:
```
python matconvnet_hr101_to_pickle.py
        --matlab_model_path /path/to/pretrained_model
        --weight_file_path  /path/to/pickle_file
```


Requirements:
- opencv-python
- dlib
- Tensorflow

