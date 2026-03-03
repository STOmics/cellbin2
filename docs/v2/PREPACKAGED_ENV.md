# cellbin2_env for Linux and Windows

The pre-packaged environments for Linux and Windows are available on the BGI cloud drive:

| Platform | File Name | Download Link | Extraction Code |
|----------|---------------------------|-----------------------------|-----------------|
| Linux | cellbin2_env_Linux.tar.gz | [Download](https://bgipan.genomics.cn/#/link/wzskohFHIlB5T2xOVD3c) | UYhj |
| Windows | cellbin2_env_Windows.tar.gz | [Download](https://bgipan.genomics.cn/#/link/mM9od4ennL22pHJNRIJc) | unO7 |


## Linux：
### After downloading the original compressed package from the BGI cloud drive, extract it to your Anaconda 'envs' folder.
**Extraction command**
```bash
#Create the target directory
mkdir -p /path/to/extract/anaconda/envs/cellbin2
#Extract the package
tar -xzvf /path/to/cellbin2_env_linux.tar.gz -C "/path/to/anaconda/envs/cellbin2"
```
**Activate the environment**
```bash
conda activate cellbin2
```
**Verify the environment**
```bash
python --version
conda list #If everything is correct, you should see all dependencies of this environment.
```

**After activation**
According to the CUDA and cuDNN versions on your Linux system, select compatible versions of torch and ONNXRuntime from:
[CUDA&cuDNN&ONNXRuntime Compatibility](https://runtime.onnx.org.cn/docs/execution-providers/CUDA-ExecutionProvider.html) 
[CUDA&cuDNN&torch Compatibility](https://pytorch.org/get-started/previous-versions/)


**Test GPU**
```bash
#Enter Python interactive shell
import onnxruntime
print(onnxruntime.get_available_providers())

#Expected result (contains CUDAExecutionProvider)
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
#Incorrect result
['CPUExecutionProvider']
```
If the expected result is returned, the environment is successfully configured. Run demo.py for further testing;
If you see the incorrect result, refer to [Using GPU README](https://github.com/STOmics/cellbin2/blob/release/docs/v2/Using_GPU_README_EN.md).


**Install the editable mode**
```bash
#Run after activating the environment
pip install -e .
#Only in editable mode can modifications to cellbin2 be synced to site-package
```


## Windows：
### After downloading the original compressed package from the BGI cloud drive, extract it to your Anaconda 'envs' folder.
**Extraction command**
```bash
#Create the target directory
mkdir -p /path/to/extract/anaconda/envs/cellbin2
#Extract the package
tar -xzvf /path/to/cellbin2_env_linux.tar.gz -C "/path/to/anaconda/envs/cellbin2"
```
**Acativate the environment**
```bash
conda activate cellbin2
```
**Verify the environment**
```bash
python --version
conda list #If everything is correct, you should see all dependencies of this environment.
```

**After activation**
According to the CUDA and cuDNN versions on your Linux system, select compatible versions of torch and ONNXRuntime from:
[CUDA&cuDNN&ONNXRuntime Compatibility](https://runtime.onnx.org.cn/docs/execution-providers/CUDA-ExecutionProvider.html) 
[CUDA&cuDNN&torch Compatibility](https://pytorch.org/get-started/previous-versions/)

**Test GPU**
```bash
#Enter Python interactive shell
import onnxruntime
print(onnxruntime.get_available_providers())

#Expected result (contains CUDAExecutionProvider)
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
#Incorrect result
['CPUExecutionProvider']
```
If the expected result is returned, the environment is successfully configured. Run demo.py for further testing;
If you see the incorrect result, refer to [Using GPU README](https://github.com/STOmics/cellbin2/blob/release/docs/v2/Using_GPU_README_EN.md).



**Install the editable mode**
```bash
#Run after activating the environment
pip install -e .
#Only in editable mode can modifications to cellbin2 be synced to site-package
```
