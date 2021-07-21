import os
import time
import tensorflow as tf
from experimental.sunkaixuan.super_resolution.dataset import DIV2K
from model.common import DATASET_DIR, SCALE, model_to_resume
from experimental.sunkaixuan.super_resolution.model.xlsr import Xlsr
from experimental.sunkaixuan.super_resolution.trainer import XlsrTrainer


valid_loader = DIV2K(scale=SCALE, downgrade='bicubic', subset='valid')
valid_ds = valid_loader.dataset(batch_size=1, random_transform=False, repeat_count=1)

xlsr = Xlsr()
checkpoint_dir = os.path.join(DATASET_DIR, f'weights/{model_to_resume}')
trainer = XlsrTrainer(model=xlsr.xlsr(num_gblocks=3,  scale=SCALE), checkpoint_dir=checkpoint_dir)

# Evaluate model on full validation set.
psnr, score, runtime = trainer.evaluate(valid_ds)
print(f'PSNR = {psnr.numpy():3f}, SCORE = {score.numpy():3f}, RUNTIME = {runtime.numpy():3f}')

saved_model_path = os.path.join(checkpoint_dir, 'saved_models', str(int(time.time())))
tf.saved_model.save(trainer.model, saved_model_path)

# Test for task 2
test_lr_files = [os.path.join(DATASET_DIR, 'images/Task2', f'Test_{image_id:02}.png') for image_id in range(10)]
test_hr_files = [os.path.join(DATASET_DIR, 'images/Task2', f'GT_{image_id:02}.png') for image_id in range(10)]
test_lr_dataset = DIV2K._images_dataset(test_lr_files)
test_hr_dataset = DIV2K._images_dataset(test_hr_files)
ds = tf.data.Dataset.zip((test_lr_dataset, test_hr_dataset)).batch(1)
psnr, score, runtime = trainer.evaluate(ds, save_result=True, checkpoint_dir=checkpoint_dir)


