# Creating a Specific Version of Python and PyTorch Environment on Colab.

This project will guide free users in creating a specific version of Python and PyTorch execution environment on Google Colab.

As proof as concept, I use Miniconda version 3.8 to create a Python 3.7 environment, which can use both PyTorch 1.7.1 and tensorflow 2.1.0 version.

Please note:

1. How to deal with Colab time limit?

    This document does not overcome the time limit issue for free users, who will need to wait for a specific period before using it again. Therefore, please make good use of Checkpoint to save and continue training progress.

2. Can I use the realtime console with the Colab cell?
    
    Unfortunately, I was not successful to do so. In this usage, I cannot utilize the cell's ability to compile and execute programs in real-time. I can only use the "activate virtual environment" -> "run file" method. This makes development and debugging more annoyed. If you have a better approach, feel free to share it with me or submit a PR. Thank you!

3. Please use the version of Miniconda installer which is higher than the Python version you wish to use to create the virtual environment.

## How to use it?

1. Clone the project
    
    ```bash
    git clone https://github.com/liuyuweitarek/downgrade-upgrade-colab-python.git custom_env_colab
    ```

2. Place the project in your Google Drive.

3. Follow the instructions in the `Creating Specific Version of Python and PyTorch Environment on Colab.ipynb` notebook.
