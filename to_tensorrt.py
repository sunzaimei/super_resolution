import logging
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from experimental.sunkaixuan.super_resolution.dataset import DIV2K, DATASET_DIR
from experimental.sunkaixuan.super_resolution.model.common import resolve, evaluate
from model.common import SCALE, model_to_resume, saved_model_to_resume

valid_loader = DIV2K(scale=SCALE, downgrade='bicubic', subset='valid', mode='valid')


def speed_test(name, model, steps=100):
    result = {}
    valid_ds = valid_loader.dataset(batch_size=1, random_transform=False, repeat_count=1)
    psnr, score, runtime = evaluate(model, valid_ds, save_result=False, checkpoint_dir=os.path.join(DATASET_DIR, f'weights/{model_to_resume}/saved_models'))
    result['psnr/dB'] = f'{psnr:3f}dB'
    result['score'] = f'{score:.3f}'
    result['runtime/ms'] = f'{runtime:.3f}ms'
    out = []
    tf.profiler.experimental.start(os.path.join(DATASET_DIR, f'weights/{model_to_resume}/profile/{name}'))
    inputs = tf.random.normal((1, 2040, 1152, 3))
    with tf.profiler.experimental.Trace('predict', step_num=1):
        for i in range(steps):
            timenow = time.time()
            output, latency = resolve(model, inputs)
            timeout = time.time() - timenow
            # Warm start 10 step
            if i > 10:
                out.append(latency)

    tf.profiler.experimental.stop()
    result['time/ms'] = f'{np.mean(out) * 1000:.3f}ms'
    return result


def optimize(saved_model_dir, output_saved_model_dir):
    conversion_params = tf.experimental.tensorrt.ConversionParams(precision_mode=trt.TrtPrecisionMode.INT8, maximum_cached_engines=1,
                                                                  is_dynamic_op=True, allow_build_at_runtime=True, use_calibration=True)
    converter = tf.experimental.tensorrt.Converter(saved_model_dir, conversion_params=conversion_params)

    def my_calibration_input_fn():
        valid_loader = DIV2K(scale=SCALE, downgrade='bicubic', subset='valid', mode='valid')
        for data in valid_loader.dataset(batch_size=1, random_transform=True, repeat_count=1).take(100):
            # tf.print(data[0].shape)
            # yield [data[0]]
            yield [tf.cast(data[0], tf.float32)]

    converter.convert(calibration_input_fn=my_calibration_input_fn)
    converter.save(output_saved_model_dir)

    return output_saved_model_dir


def convert_trt(saved_model_dir, output_saved_model_dir=''):
    assert os.path.exists(saved_model_dir), f"Input dir '{saved_model_dir}' does not exist"
    assert os.path.isdir(saved_model_dir), f"Input dir '{saved_model_dir}' is not a directory"

    # Create saved_model out dir
    if output_saved_model_dir == '':
        parent_dir = os.path.dirname(saved_model_dir)
        basename = os.path.basename(saved_model_dir) + '_trt'
        output_saved_model_dir = os.path.join(parent_dir, basename)
        logging.info(f'saved_model_dir_out will be {output_saved_model_dir}')

    return optimize(saved_model_dir, output_saved_model_dir)

if __name__ == '__main__':
    saved_model_dir = os.path.join(DATASET_DIR, f'weights/{model_to_resume}/saved_models/{saved_model_to_resume}')
    # optim_saved_model_dir = os.path.join(DATASET_DIR, f'weights/{model_to_resume}/saved_models/{saved_model_to_resume}_trt')
    optim_saved_model_dir = convert_trt(saved_model_dir=saved_model_dir, output_saved_model_dir='', )
    print(optim_saved_model_dir)
    print("conver model done... Nowing running test")
    out = {}
    model = tf.saved_model.load(saved_model_dir)
    out['original'] = speed_test('original', model, steps=100)
    model_int8 = tf.saved_model.load(optim_saved_model_dir)
    out['int8'] = speed_test("int8", model_int8, steps=100)
    print("profile", out)
