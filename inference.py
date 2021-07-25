import os
import tensorflow as tf
from dataset import DIV2K
from settings import DATASET_DIR, SCALE, model_to_resume
from model.xlsr import Xlsr
from trainer import XlsrTrainer

valid_loader = DIV2K(scale=SCALE, downgrade='bicubic', subset='valid')
valid_ds = valid_loader.dataset(batch_size=1, random_transform=False, repeat_count=1)

checkpoint_dir = os.path.join(DATASET_DIR, f'weights/{model_to_resume}')
# saved_model_path = os.path.join(checkpoint_dir, 'saved_models', saved_model_to_resume)
# imported = tf.saved_model.load(saved_model_path)
# imported.summary()

xlsr = Xlsr()
model = xlsr.xlsr(num_gblocks=3, scale=SCALE)
trainer = XlsrTrainer(model=model, checkpoint_dir=checkpoint_dir)
# trainer.restore()

# Evaluate model on full validation set.
psnr, score, runtime = trainer.evaluate(valid_ds, save_result=True, checkpoint_dir=checkpoint_dir)
print(f'PSNR = {psnr.numpy():3f}, SCORE = {score.numpy():3f}, RUNTIME = {runtime.numpy():3f}')


# Test for task 2
if SCALE == 4:
    test_lr_files = [os.path.join(DATASET_DIR, 'images/Task2', f'Test_{image_id:02}.png') for image_id in range(10)]
    test_hr_files = [os.path.join(DATASET_DIR, 'images/Task2', f'GT_{image_id:02}.png') for image_id in range(10)]
    test_lr_dataset = DIV2K._images_dataset(test_lr_files)
    test_hr_dataset = DIV2K._images_dataset(test_hr_files)
    ds = tf.data.Dataset.zip((test_lr_dataset, test_hr_dataset, tf.data.Dataset.from_tensor_slices(test_lr_files))).batch(1)
    psnr, score, runtime = trainer.evaluate(ds, save_result=True, checkpoint_dir=checkpoint_dir)
    print(f"task 2 psnr is {psnr}")
