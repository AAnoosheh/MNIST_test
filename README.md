
# MNIST Training and Inference Pipeline


## Prerequisites
- Python 3
- PyTorch 0.4+
- Torch-vision


### Training & Testing

- Train a model:
```
python train.py  --name <experiment_name>
```
MNIST dataset will be automatically downloaded into `./datasets/mnist` by default, if not already.
Checkpoints will be saved by default to `./checkpoints/<experiment_name>/`

- Resume training:
```
python train.py  --name <experiment_name> --continue_train --which_epoch <N>
```
Where `N` is an epoch number for which a checkpoint was saved.

- Test the model:
```
python test.py  --name <experiment_name> --which_epoch <N>
```
The script will continuously prompt for user input of paths pointing to images to be inferred one-by-one.


## Options
- Flags: See `options/train_options.py` for training-specific flags and their default values; see `options/test_options.py` for test-specific flags; and see `options/base_options.py` for all common flags.

- Hyperparameter tuning: The aforementioned flags include `--num_epochs`, `--kernel_width`, `--batch_size`, `--lr`, and `--momentum`, all with default values set.

- CPU/GPU (default `--gpu_id 0`): set`--gpu_id -1` to use CPU mode.

- Preprocessing: Optional horizontal image flipping during training, disabled using the `--no_flip` flag.  Set `--n_threads` to change number of data-loading threads.

- Options used for each training run are also saved to `./checkpoints/<experiment_name>/opt.txt` for future reference

- Altering dataset requires modifying the main scripts and models. Altering loss function and optimizer can be done manually within train.py
