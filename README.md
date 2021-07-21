# super_resolution

report analysis is here: https://docs.google.com/presentation/d/1mRGRsFEkz3rmhFULSEvYm2XQU42ZaSXVJOvnAdOoVAA/edit?usp=sharing

Make sure D2IK dataset is downloaded and put at a certain location. Modify DATASET_DIR in common.py to indicate that. 
- For training, simply run 'train.py'. For different scale level, simply change SCALE in common.py
- For inference, run 'inference.py'. First make sure model_to_resume variable in common.py contains the saved model directory.
- For int8 quantization, network speed profile, speed and accuracy comparison before and after quantization, run 'to_tensorrt.py'. 
For this need to make sure both model_to_resume, saved_model_to_resume refers to the correct directory
