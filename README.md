# super_resolution

## This repo aims to implement the super resolution algorithm from paper, and evaluate results on various dataset.

report analysis is here: https://docs.google.com/presentation/d/1mRGRsFEkz3rmhFULSEvYm2XQU42ZaSXVJOvnAdOoVAA/edit?usp=sharing

Pre-request: tensorflow-gpu 2.4, tensorflow_model_optimization 0.6.0
Make sure D2IK dataset is downloaded and put at a certain location. Modify DATASET_DIR in config.py to indicate that. 

- For training, simply run 'train.py'. For different scale level, simply change SCALE in config.py
- For quantization aware training, also run 'train.py', but change 'quantization' in config.py to True
- For inference, run 'inference.py'. First make sure model_to_resume variable in config.py contains the saved model directory. 
- For post training quantization, network speed profile, speed and accuracy comparison before and after quantization, run 'to_tensorrt.py'. For this need to make sure both model_to_resume, saved_model_to_resume refers to the correct directory
