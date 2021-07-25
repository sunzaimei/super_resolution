import time
import numpy as np
from dataset import DIV2K
from model.xlsr import Xlsr
from trainer import XlsrTrainer
import os
from settings import DATASET_DIR, model_to_resume, SCALE
import tensorflow as tf
import pathlib

valid_loader = DIV2K(scale=SCALE, downgrade='bicubic', subset='valid', mode='valid')
valid_ds = valid_loader.dataset(batch_size=16, random_transform=True, repeat_count=1)

checkpoint_dir = os.path.join(DATASET_DIR, f'weights/{model_to_resume}')
xlsr = Xlsr()
trainer = XlsrTrainer(model=xlsr.xlsr(num_gblocks=3, scale=SCALE), checkpoint_dir=checkpoint_dir)
trainer.restore()
# save model to pb
model_version = str(int(time.time()))
export_dir_base = os.path.join(checkpoint_dir, 'saved_models', model_version)

psnr, score, runtime = trainer.evaluate(valid_ds)
print(f'PSNR = {psnr.numpy():3f}, SCORE = {score.numpy():3f}, RUNTIME = {runtime.numpy():3f}')
tf.saved_model.save(trainer.model, export_dir_base)

print("start quantizatino")
# quantization
tflite_models_dir = pathlib.Path(os.path.join(checkpoint_dir, "saved_models"))
tflite_models_dir.mkdir(exist_ok=True, parents=True)
# converter = tf.lite.TFLiteConverter.from_keras_model(trainer.model)
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir_base)
tflite_model = converter.convert()
# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"model.tflite"
tflite_model_file.write_bytes(tflite_model)

def representative_data_gen():
    for data in valid_loader.dataset(batch_size=1, random_transform=True, repeat_count=1).take(100):
        # tf.print(data[0].shape)
        yield [tf.cast(data[0], tf.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model(export_dir_base)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_data_gen
# # Ensure that if any ops can't be quantized, the converter throws an error
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# # Set the input and output tensors to uint8 (APIs added in r2.3)
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# converter.allow_custom_ops = True
# converter.experimental_new_converter = False
tflite_model_quant = converter.convert()


# Save the quantized model:
tflite_model_quant_file = tflite_models_dir/"mnist_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)


def run_tflite_model(tflite_file):
    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    for data in valid_loader.dataset(batch_size=1, random_transform=True, repeat_count=1).take(1):
        test_image = data[0]
        tf.print(test_image.shape)
    # Check if the input type is quantized, then rescale input data to uint8
    print("input detalis dtype", input_details['dtype'], input_details["quantization"])
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point

    # test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    test_image = test_image.numpy().astype(input_details["dtype"])
    tf.print("input_details", input_details["index"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    print(output)
    return output
#
run_tflite_model(tflite_model_quant_file)