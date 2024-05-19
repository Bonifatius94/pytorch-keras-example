# PyTorch Keras Example

## About
This repository outlines how to use PyTorch as backend for the Keras 3.0 API.

As PyTorch has a painless GPU setup for AI trainings, it might be interesting
to use PyTorch under the hood, but with the familiar syntax from TensorFlow Keras.

## GPU Setup
First, install a proprietary NVIDIA driver for your GPU (version 535 in this case).

```sh
sudo apt-get update && sudo apt-get install -y nvidia-driver-535 nvidia-dkms-535
```

GPU driver changes usually require a reboot to take effect.

```sh
sudo reboot
```

According to the "getting started" [page](https://pytorch.org/get-started/locally/)
of PyTorch, execute following command to set up PyTorch to use GPUs.

```sh
python -m pip install torch torchvision torchaudio
```

Additionally, install the Keras Core 3.0 package to make use of Keras in PyTorch.
Unlike TensorFlow, Keras is not automatically shipped with PyTorch. It needs to be
installed separately.

```sh
python -m pip install keras-core
```

## Training
Now, it's time to check if the setup was successful. Open a console to monitor the GPU.

```sh
watch nvidia-smi
```

Launch another console to run the training script and see if it utilizes the GPU as expected.

```sh
python keras.py
```

Outputs should be something like this:

```text
Using PyTorch backend.
Epoch 1/10
938/938 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - loss: 0.4897 - sparse_categorical_accuracy: 0.8403 - val_loss: 0.0517 - val_sparse_categorical_accuracy: 0.9824
Epoch 2/10
938/938 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - loss: 0.0926 - sparse_categorical_accuracy: 0.9717 - val_loss: 0.0438 - val_sparse_categorical_accuracy: 0.9863
Epoch 3/10
938/938 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - loss: 0.0756 - sparse_categorical_accuracy: 0.9766 - val_loss: 0.0373 - val_sparse_categorical_accuracy: 0.9863
Epoch 4/10
938/938 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - loss: 0.0628 - sparse_categorical_accuracy: 0.9808 - val_loss: 0.0334 - val_sparse_categorical_accuracy: 0.9876
Epoch 5/10
938/938 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - loss: 0.0520 - sparse_categorical_accuracy: 0.9835 - val_loss: 0.0252 - val_sparse_categorical_accuracy: 0.9913
Epoch 6/10
938/938 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - loss: 0.0498 - sparse_categorical_accuracy: 0.9854 - val_loss: 0.0304 - val_sparse_categorical_accuracy: 0.9897
Epoch 7/10
938/938 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - loss: 0.0499 - sparse_categorical_accuracy: 0.9848 - val_loss: 0.0263 - val_sparse_categorical_accuracy: 0.9913
Epoch 8/10
938/938 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - loss: 0.0396 - sparse_categorical_accuracy: 0.9875 - val_loss: 0.0279 - val_sparse_categorical_accuracy: 0.9906
Epoch 9/10
938/938 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - loss: 0.0392 - sparse_categorical_accuracy: 0.9877 - val_loss: 0.0242 - val_sparse_categorical_accuracy: 0.9914
Epoch 10/10
938/938 ━━━━━━━━━━━━━━━━━━━━ 4s 4ms/step - loss: 0.0366 - sparse_categorical_accuracy: 0.9882 - val_loss: 0.0286 - val_sparse_categorical_accuracy: 0.9914
```

```text
Sat Feb 24 21:08:26 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.154.05             Driver Version: 535.154.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3060        Off | 00000000:01:00.0 Off |                  N/A |
|  0%   34C    P2              63W / 170W |    216MiB / 12288MiB |     39%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A     38167      C   python                                      210MiB |
+---------------------------------------------------------------------------------------+
```

In this case NVIDIA schedules a compute task (Type C) for our python training with 39% GPU utilization.
