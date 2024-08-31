# Creating a Specific Version of Python and PyTorch Environment on Colab

This document will guide free users in creating a specific version of Python, PyTorch and Tensorflow execution environment on Google Colab. 

Based on the steps provided here, you can create any Python(version >= 3.7) environment in Google Colab. If you're using TensorFlow with GPU support requires specific actions and version compatibility. 
Please refer to [this table](https://www.tensorflow.org/install/source#gpu) for the corresponding versions.

**Noted that this method has limitations.** 

With this method, cell cannot directly execute Python code from the environment you create. You need to activate the created environment at the start of each cell execution, which is equivalent to running your command in the terminal.

i.e. Cell will be like below to execute your code:

```
%%shell
eval "$(conda shell.bash hook)"
conda activate myenv
python main.py
```

As proof of concept, I use Miniconda version 3.8 to create a Python 3.7 environment, which can use both PyTorch 1.7.1 and tensorflow 2.1.0 version here.

If have a better approach, feel free to share it with me or submit a PR. Thank you!

Furthermore,

1. How to deal with Colab time limit?
  
    This document does not overcome the time limit issue for free users, who will need to wait for a specific period before using it again. Therefore, please **make good use of Checkpoint to save and continue training progress**.
 
2. Please use the version of Miniconda installer which is higher than the Python version you wish to use to create the virtual environment.
