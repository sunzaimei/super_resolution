import datetime
import time
from model.common import evaluate_quantized_model
from dataset import DIV2K
from model.xlsr import Xlsr
from trainer import XlsrTrainer
import os
from settings import DATASET_DIR, SCALE, quantization
import tensorflow as tf
import tensorflow_model_optimization as tfmot


def apply_quantization_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer


train_loader = DIV2K(scale=SCALE, downgrade='bicubic', subset='train', mode='train')
valid_loader = DIV2K(scale=SCALE, downgrade='bicubic', subset='train', mode='valid')

if SCALE == 4:
    add_noise = True
elif SCALE == 3:
    add_noise = False
# Create corresponding tf.data.Dataset
train_ds = train_loader.dataset(batch_size=16, random_transform=True, repeat_count=-1, add_noise=add_noise)
valid_ds = valid_loader.dataset(batch_size=16, random_transform=True, repeat_count=1, add_noise=add_noise)

xlsr = Xlsr()
model = xlsr.xlsr(num_gblocks=3, scale=SCALE)
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if quantization:
    checkpoint_dir = os.path.join(DATASET_DIR, f'weights/xlsr-16-x{SCALE}_{current_time}_quantization')
    annotated_model = tf.keras.models.clone_model(model, clone_function=apply_quantization_to_dense)
    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    trainer = XlsrTrainer(model=quant_aware_model, checkpoint_dir=checkpoint_dir)
else:
    checkpoint_dir = os.path.join(DATASET_DIR, f'weights/xlsr-16-x{SCALE}_{current_time}')
    trainer = XlsrTrainer(model=model, checkpoint_dir=checkpoint_dir)
print(f'Model will be stored in checkpoint_dir: {checkpoint_dir}')

# trainer.train(train_ds, valid_ds, steps=2000, evaluate_every=100, save_best_only=True)
trainer.train(train_ds, valid_ds, steps=500000, evaluate_every=100, save_best_only=True)

# Restore from checkpoint with highest PSNR.
trainer.restore()
# Evaluate model on full validation set.
psnr_v, score, runtime = trainer.evaluate(valid_ds)
print(f'PSNR = {psnr_v.numpy():3f}, SCORE = {score.numpy():3f}, RUNTIME = {runtime.numpy():3f}')

# Save model
saved_model_path = os.path.join(checkpoint_dir, 'saved_models', str(int(time.time())))
tf.saved_model.save(trainer.model, saved_model_path)
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

# Save h5
keras_model_file = os.path.join(checkpoint_dir, 'saved_models', 'model.h5')
trainer.model.save(keras_model_file)

if quantization:
    # `quantize_scope` is needed for deserializing HDF5 models.
    with tfmot.quantization.keras.quantize_scope():
      loaded_model = tf.keras.models.load_model(keras_model_file)
    loaded_model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()
    with open(os.path.join(checkpoint_dir, 'saved_models', 'model.tflite'), 'wb') as f:
        f.write(quantized_tflite_model)

    valid_loader = DIV2K(scale=SCALE, downgrade='bicubic', subset='valid')
    test_accuracy = evaluate_quantized_model(quantized_tflite_model, checkpoint_dir, valid_loader)
    print('Quant TFLite test_accuracy:', test_accuracy)


