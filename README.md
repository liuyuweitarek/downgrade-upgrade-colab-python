# Creating a Specific Version of Python and PyTorch Environment on Colab

This document will guide free users in creating a specific version of Python, PyTorch and Tensorflow execution environment on Google Colab. 

Based on the steps provided here, you can create any Python(version >= 3.7) environment in Google Colab. 

**Noted that this method has limitations.** 

With this method, cell cannot directly execute Python code from the environment you create. You need to activate the created environment at the start of each cell execution, which is equivalent to running your command in the terminal.

i.e. Cell will be like below to execute your code:

```
%%shell
eval "$(conda shell.bash hook)"
conda activate myenv
python main.py
```

## Prerequisite

### 1. Check CUDA versions
  Please refer to [this table](https://www.tensorflow.org/install/source?hl=zh-tw#gpu) to find the Python Version you're using and the corresponding supported CUDA Version. 

P.S. No matter you are using PyTorch or TensorFlow.
 
### 2. Check your package version 

#### For Pytorch

  Check the table below, and look for whether the package is in this [package source list](https://download.pytorch.org/whl/torch/).

  For example:
  - Python 3.7(cp37)
  - PyTorch-1.7.1
  - CUDA Version 10.1(cu101)
  - and google colab is linux based system(linux_x86_64).
  
  I then should found `torch-1.7.1+cu101-cp37-cp37m-linux_x86_64.whl` in the list, which cound be installed in command like:
    
  ```bash
  $ python -m pip install torch==1.7.1+cu101 --extra-index-url https://download.pytorch.org/whl --no-cache-dir
  ```

<table>
  <thead>
    <tr>
      <th>Torch Version</th>
      <th>Available CUDA Version</th>
      <th>Python Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2.0.1</td>
      <td>cu117, cu118</td>
      <td>cp38, cp39, cp310, cp311</td>
    </tr>
    <tr>
      <td>2.0.0</td>
      <td>cu117, cu118</td>
      <td>cp38, cp39, cp310, cp311</td>
    </tr>
    <tr>
      <td>1.13.1</td>
      <td>cu116, cu117</td>
      <td>cp37, cp38, cp39, cp310</td>
    </tr>
    <tr>
      <td>1.13.0</td>
      <td>cu116, cu117</td>
      <td>cp37, cp38, cp39, cp310</td>
    </tr>
    <tr>
      <td>1.12.1</td>
      <td>cu113, cu116</td>
      <td>cp37, cp38, cp39, cp310</td>
    </tr>
    <tr>
      <td>1.12.0</td>
      <td>cu113, cu116</td>
      <td>cp37, cp38, cp39, cp310</td>
    </tr>
    <tr>
      <td>1.11.0</td>
      <td>cu113, cu115</td>
      <td>cp37, cp38, cp39, cp310</td>
    </tr>
    <tr>
      <td>1.10.2</td>
      <td>cu102, cu111, cu113</td>
      <td>cp36, cp37, cp38, cp39</td>
    </tr>
    <tr>
      <td>1.10.1</td>
      <td>cu102, cu111, cu113</td>
      <td>cp36, cp37, cp38, cp39</td>
    </tr>
    <tr>
      <td>1.10.0</td>
      <td>cu102, cu111, cu113</td>
      <td>cp36, cp37, cp38, cp39</td>
    </tr>
    <tr>
      <td>1.9.1</td>
      <td>cu102, cu111</td>
      <td>cp36, cp37, cp38, cp39</td>
    </tr>
    <tr>
      <td>1.9.0</td>
      <td>cu102, cu111</td>
      <td>cp36, cp37, cp38, cp39</td>
    </tr>
    <tr>
      <td>1.8.1</td>
      <td>cu101, cu102, cu111</td>
      <td>cp36, cp37, cp38, cp39</td>
    </tr>
    <tr>
      <td>1.8.0</td>
      <td>cu101, cu111</td>
      <td>cp36, cp37, cp38, cp39</td>
    </tr>
    <tr>
      <td>1.7.1</td>
      <td>cu101, cu110</td>
      <td>cp36, cp37, cp38, cp39</td>
    </tr>
    <tr>
      <td>1.7.0</td>
      <td>cu101, cu110</td>
      <td>cp36, cp37, cp38</td>
    </tr>
    <tr>
      <td>1.6.0</td>
      <td>cu101</td>
      <td>cp36, cp37, cp38</td>
    </tr>
    <tr>
      <td>1.5.1</td>
      <td>cu92, cu101</td>
      <td>cp35, cp36, cp37, cp38</td>
    </tr>
    <tr>
      <td>1.5.0</td>
      <td>cu92, cu101</td>
      <td>cp35, cp36, cp37, cp38</td>
    </tr>
    <tr>
      <td>1.4.0</td>
      <td>cu92</td>
      <td>cp35, cp36, cp37, cp38</td>
    </tr>
    <tr>
      <td>1.3.1</td>
      <td>cu92</td>
      <td>cp35, cp36, cp37</td>
    </tr>
    <tr>
      <td>1.3.0</td>
      <td>cu92</td>
      <td>cp35, cp36, cp37</td>
    </tr>
    <tr>
      <td>1.2.0</td>
      <td>cu92</td>
      <td>cp35, cp36, cp37</td>
    </tr>
  </tbody>
</table>

#### For Tensorflow

Check out [the version table](https://www.tensorflow.org/install/source?hl=zh-tw#gpu). 

For example,
- Python 3.7
- Tensorflow 2.1.0

At `requirements.txt`,

```
tensorflow-gpu==2.1.0
protobuf==3.20.1
```

### 3. Install cudnn + CUDA

Based on the `CUDA Version` you determined above, find the cudnn package to install:

Below are the download links for CUDA versions <= 11. 

|Cudnn Version|Sources|
| --- | --- |
|v5 - v8.8.0| https://developer.download.nvidia.com/compute/redist/cudnn |
|v8.4.0 - v9.3.0| https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64 |

If you are using CUDA version >= 12, you might consider downloading it directly from the official website.

For example:

- CUDA 10.1

I choose `cudnn-10.1-linux-x64-v7.6.5.32.tgz` to download.

<table><thead><tr><th>cuDNN</th><th>CUDA</th></tr></thead><tbody><tr><td>cuDNN v8.4.0 (April 1st, 2022)</td><td>CUDA 11.x</td></tr><tr><td>cuDNN v8.4.0 (April 1st, 2022)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v8.3.3 (March 18th, 2022)</td><td>CUDA 11.5</td></tr><tr><td>cuDNN v8.3.3 (March 18th, 2022)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v8.3.2 (January 10th, 2022)</td><td>CUDA 11.5</td></tr><tr><td>cuDNN v8.3.2 (January 10th, 2022)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v8.3.1 (November 22nd, 2021)</td><td>CUDA 11.5</td></tr><tr><td>cuDNN v8.3.1 (November 22nd, 2021)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v8.3.0 (November 3rd, 2021)</td><td>CUDA 11.5</td></tr><tr><td>cuDNN v8.3.0 (November 3rd, 2021)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v8.2.4 (September 2nd, 2021)</td><td>CUDA 11.4</td></tr><tr><td>cuDNN v8.2.4 (September 2nd, 2021)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v8.2.2 (July 6th, 2021)</td><td>CUDA 11.4</td></tr><tr><td>cuDNN v8.2.2 (July 6th, 2021)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v8.2.1 (June 7th, 2021)</td><td>CUDA 11.x</td></tr><tr><td>cuDNN v8.2.1 (June 7th, 2021)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v8.2.0 (April 23rd, 2021)</td><td>CUDA 11.x</td></tr><tr><td>cuDNN v8.2.0 (April 23rd, 2021)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v8.1.1 (Feburary 26th, 2021)</td><td>CUDA 11.0,11.1 and 11.2</td></tr><tr><td>cuDNN v8.1.1 (Feburary 26th, 2021)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v8.1.0 (January 26th, 2021)</td><td>CUDA 11.0,11.1 and 11.2</td></tr><tr><td>cuDNN v8.1.0 (January 26th, 2021)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v8.0.5 (November 9th, 2020)</td><td>CUDA 11.1</td></tr><tr><td>cuDNN v8.0.5 (November 9th, 2020)</td><td>CUDA 11.0</td></tr><tr><td>cuDNN v8.0.5 (November 9th, 2020)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v8.0.5 (November 9th, 2020)</td><td>CUDA 10.1</td></tr><tr><td>cuDNN v8.0.4 (September 28th, 2020)</td><td>CUDA 11.1</td></tr><tr><td>cuDNN v8.0.4 (September 28th, 2020)</td><td>CUDA 11.0</td></tr><tr><td>cuDNN v8.0.4 (September 28th, 2020)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v8.0.4 (September 28th, 2020)</td><td>CUDA 10.1</td></tr><tr><td>cuDNN v8.0.3 (August 26th, 2020)</td><td>CUDA 11.0</td></tr><tr><td>cuDNN v8.0.3 (August 26th, 2020)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v8.0.3 (August 26th, 2020)</td><td>CUDA 10.1</td></tr><tr><td>cuDNN v8.0.2 (July 24th, 2020)</td><td>CUDA 11.0</td></tr><tr><td>cuDNN v8.0.2 (July 24th, 2020)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v8.0.2 (July 24th, 2020)</td><td>CUDA 10.1</td></tr><tr><td>cuDNN v8.0.1 RC2 (June 26th, 2020)</td><td>CUDA 11.0</td></tr><tr><td>cuDNN v8.0.1 RC2 (June 26th, 2020)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v7.6.5 (November 18th, 2019)</td><td>CUDA 10.2</td></tr><tr><td>cuDNN v7.6.5 (November 5th, 2019)</td><td>CUDA 10.1</td></tr><tr><td>cuDNN v7.6.5 (November 5th, 2019)</td><td>CUDA 10.0</td></tr><tr><td>cuDNN v7.6.5 (November 5th, 2019)</td><td>CUDA 9.2</td></tr><tr><td>cuDNN v7.6.5 (November 5th, 2019)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v7.6.4 (September 27, 2019)</td><td>CUDA 10.1</td></tr><tr><td>cuDNN v7.6.4 (September 27, 2019)</td><td>CUDA 10.0</td></tr><tr><td>cuDNN v7.6.4 (September 27, 2019)</td><td>CUDA 9.2</td></tr><tr><td>cuDNN v7.6.4 (September 27, 2019)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v7.6.3 (August 23, 2019)</td><td>CUDA 10.1</td></tr><tr><td>cuDNN v7.6.3 (August 23, 2019)</td><td>CUDA 10.0</td></tr><tr><td>cuDNN v7.6.3 (August 23, 2019)</td><td>CUDA 9.2</td></tr><tr><td>cuDNN v7.6.3 (August 23, 2019)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v7.6.2 (July 22, 2019)</td><td>CUDA 10.1</td></tr><tr><td>cuDNN v7.6.2 (July 22, 2019)</td><td>CUDA 10.0</td></tr><tr><td>cuDNN v7.6.2 (July 22, 2019)</td><td>CUDA 9.2</td></tr><tr><td>cuDNN v7.6.2 (July 22, 2019)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v7.6.1 (June 24, 2019)</td><td>CUDA 10.1</td></tr><tr><td>cuDNN v7.6.1 (June 24, 2019)</td><td>CUDA 10.0</td></tr><tr><td>cuDNN v7.6.1 (June 24, 2019)</td><td>CUDA 9.2</td></tr><tr><td>cuDNN v7.6.1 (June 24, 2019)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v7.6.0 (May 20, 2019)</td><td>CUDA 10.1</td></tr><tr><td>cuDNN v7.6.0 (May 20, 2019)</td><td>CUDA 10.0</td></tr><tr><td>cuDNN v7.6.0 (May 20, 2019)</td><td>CUDA 9.2</td></tr><tr><td>cuDNN v7.6.0 (May 20, 2019)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v7.5.1 (April 22, 2019)</td><td>CUDA 10.1</td></tr><tr><td>cuDNN v7.5.1 (April 22, 2019)</td><td>CUDA 10.0</td></tr><tr><td>cuDNN v7.5.1 (April 22, 2019)</td><td>CUDA 9.2</td></tr><tr><td>cuDNN v7.5.1 (April 22, 2019)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v7.5.0 (Feb 25, 2019)</td><td>CUDA 10.1</td></tr><tr><td>cuDNN v7.5.0 (Feb 21, 2019)</td><td>CUDA 10.0</td></tr><tr><td>cuDNN v7.5.0 (Feb 21, 2019)</td><td>CUDA 9.2</td></tr><tr><td>cuDNN v7.5.0 (Feb 21, 2019)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v7.4.2 (Dec 14, 2018)</td><td>CUDA 10.0</td></tr><tr><td>cuDNN v7.4.2 (Dec 14, 2018)</td><td>CUDA 9.2</td></tr><tr><td>cuDNN v7.4.2 (Dec 14, 2018)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v7.4.1 (Nov 8, 2018)</td><td>CUDA 10.0</td></tr><tr><td>cuDNN v7.4.1 (Nov 8, 2018)</td><td>CUDA 9.2</td></tr><tr><td>cuDNN v7.4.1 (Nov 8, 2018)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v7.3.1 (Sept 28, 2018)</td><td>CUDA 10.0</td></tr><tr><td>cuDNN v7.3.1 (Sept 28, 2018)</td><td>CUDA 9.2</td></tr><tr><td>cuDNN v7.3.1 (Sept 28, 2018)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v7.3.0 (Sept 19, 2018)</td><td>CUDA 10.0</td></tr><tr><td>cuDNN v7.3.0 (Sept 19, 2018)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v7.2.1 (August 7, 2018)</td><td>CUDA 9.2</td></tr><tr><td>cuDNN v7.1.4 (May 16, 2018)</td><td>CUDA 9.2</td></tr><tr><td>cuDNN v7.1.4 (May 16, 2018)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v7.1.4 (May 16, 2018)</td><td>CUDA 8.0</td></tr><tr><td>cuDNN v7.1.3 (April 17, 2018)</td><td>CUDA 9.1</td></tr><tr><td>cuDNN v7.1.3 (April 17, 2018)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v7.1.3 (April 17, 2018)</td><td>CUDA 8.0</td></tr><tr><td>cuDNN v7.1.2 (Mar 21, 2018)</td><td>CUDA 9.1 &amp; 9.2</td></tr><tr><td>cuDNN v7.1.2 (Mar 21, 2018)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v7.0.5 (Dec 11, 2017)</td><td>CUDA 9.1</td></tr><tr><td>cuDNN v7.0.5 (Dec 5, 2017)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v7.0.5 (Dec 5, 2017)</td><td>CUDA 8.0</td></tr><tr><td>cuDNN v7.0.4 (Nov 13, 2017)</td><td>CUDA 9.0</td></tr><tr><td>cuDNN v6.0 (April 27, 2017)</td><td>CUDA 8.0</td></tr><tr><td>cuDNN v6.0 (April 27, 2017)</td><td>CUDA 7.5</td></tr><tr><td>cuDNN v5.1 (Jan 20, 2017)</td><td>CUDA 8.0</td></tr><tr><td>cuDNN v5.1 (Jan 20, 2017)</td><td>CUDA 7.5</td></tr><tr><td>cuDNN v5 (May 27, 2016)</td><td>CUDA 8.0</td></tr><tr><td>cuDNN v5 (May 12, 2016)</td><td>CUDA 7.5</td></tr><tr><td>cuDNN v4 (Feb 10, 2016)</td><td>CUDA 7.0 and later.</td></tr><tr><td>cuDNN v3 (September 8, 2015)</td><td>CUDA 7.0 and later.</td></tr><tr><td>cuDNN v2 (March 17,2015)</td><td>CUDA 6.5 and later.</td></tr><tr><td>cuDNN v1 (cuDNN 6.5 R1)</td><td></td></tr></tbody></table>

Furthermore,

1. How to deal with Colab time limit?
  
    This document does not overcome the time limit issue for free users, who will need to wait for a specific period before using it again. Therefore, please **make good use of Checkpoint to save and continue training progress**.
 
2. Please use the version of Miniconda installer which is higher than the Python version you wish to use to create the virtual environment.
