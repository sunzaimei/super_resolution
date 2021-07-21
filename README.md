# super_resolution

report analysis is here: https://docs.google.com/presentation/d/1mRGRsFEkz3rmhFULSEvYm2XQU42ZaSXVJOvnAdOoVAA/edit?usp=sharing

Make sure D2IK dataset is downloaded and put at a certain location. Modify the DATASET_DIR. 
- For training, simply run train.py. For different scale level, simply change SCALE.
- For inference, run inference.py. 
- For int8 quantization, network speed profile, speed and accuracy comparison before and after quantization, run to_tensorrt.py
