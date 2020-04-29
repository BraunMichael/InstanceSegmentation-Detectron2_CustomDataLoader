# InstanceSegmentation-Detectron2
Instance Segmentation with Facebook Detectron2 (https://github.com/facebookresearch/detectron2)

# OS Requirements
Detectron only works on linux, or online notebooks like Google Colab. If installing a local linux distro, Ubuntu 20.04 is recommended: https://releases.ubuntu.com/20.04/

# Linux Installation
After setting up a new linux install (or ubuntu on windows install), get PyCharm.
```
sudo snap install pycharm-community --classic
```
Start a new project with a new virtual environment. From within pycharm, use the terminal tab at the very bottom. This should be activated in the virtual environment already, but not in python yet. Check your python version:

```
python --version
```
Then install the corresponding python-dev package, at the time of writing 20.04 comes with 3.8. Also make sure some other compiling tools are available and get xclip and xsel for kivy:

```
sudo apt-get install python3-distutils git build-essential python3.8-dev python3-tk xclip xsel
```

Pycharm sometimes has a Distutils issue from: https://stackoverflow.com/questions/55749206/modulenotfounderror-no-module-named-distutils-core

## Installing Detectron

Make sure you are in your PyCharm virtual environment, this is mostly from here: https://detectron2.readthedocs.io/tutorials/install.html

```
sudo apt install nvidia-cuda-toolkit
nvcc --version  # Check cuda version, hopefully 10.1 (or newer)
```

Install the required python packages:
```
pip install --upgrade pip
pip install opencv-python matplotlib scikit-image numpy cython Pillow imgaug imagecorruptions imageio ttictoc multiprocess lmfit joblib pyyaml==5.1
```
Currently using kivy for the gui interactions, kivy has only early support for Python 3.8 so far. for Python <3.8:
```
pip install kivy
```
On Python 3.8:
```
pip install kivy[base] kivy_examples --pre --extra-index-url https://kivy.org/downloads/simple/
```
Followed by:
```
pip install kivymd
```

You need to have all matching cuda versions, written with Cuda 10.1 as target. As of writing the following doesn't work for the prebuilt detectron2:
```diff
# DO NOT USE THIS
- pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

Instead use an older version of torch and torchvision:
```
pip install -U torch==1.4 torchvision==0.5 -f https://download.pytorch.org/whl/cu101/torch_stable.html
```
Install detectron2
```
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
Now you should be able to run the ExampleUsage.py and ExampleTrainNewData.py, Each of those scripts download their example data from within each script



# Troubleshooting
First, go to the detectron troubleshooting: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md#common-installation-issues

Graphics drivers can cause issues as well, get Nvidia graphics driver directly from site here: 
https://www.nvidia.com/Download/index.aspx


### Cuda installation troubleshooting that might help:
https://developer.nvidia.com/cuda-10.1-download-archive-base
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation

https://developer.nvidia.com/cuda-10.1-download-archive-update2?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```
