import os
from dataset import DIV2K
from model.common import evaluate_quantized_model
from config import DATASET_DIR, SCALE, model_to_resume

checkpoint_dir = os.path.join(DATASET_DIR, f'weights/{model_to_resume}')
tflite_path = os.path.join(checkpoint_dir, 'saved_models', 'model.tflite')
valid_loader = DIV2K(scale=SCALE, downgrade='bicubic', subset='valid')
test_accuracy = evaluate_quantized_model(tflite_path, checkpoint_dir, valid_loader)
print('Quant TFLite test_accuracy:', test_accuracy)
