import sys
import os
import argparse

# 1. Verify in advance whether onnxruntime is installed
try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime is not installed! Please install onnxruntime-gpu first.")
    sys.exit(1)

import requests
from tqdm import tqdm

def get_cuda_version():
    # Try nvcc nvidia
    try:
        result = os.popen('nvcc --version').read()
        for line in result.split('\n'):
            if 'release' in line:
                return line.strip()
    except Exception:
        pass
    return "Unknown"


def get_nvidia_smi_cuda_version():
    # Try nvidia-smi to get maximum supported CUDA version
    try:
        result = os.popen('nvidia-smi').read()
        for line in result.split('\n'):
            if 'CUDA Version:' in line:
                return line.strip()
    except Exception:
        pass
    return "Unknown"


def get_onnxruntime_version():
    # Try onnxruntime-gpu version
    ret = ''
    try:
        result = os.popen('pip show onnxruntime-gpu').read()
        for line in result.split('\n'):
            if 'Version:' in line:
                ret = ret + line.strip()
            if 'Required-by:' in line:
                ret = ret + line.strip()
        return ret
    except Exception:
        pass
    return "Unknown"

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
CELLBIN2_DIR = os.path.join(CURR_DIR, '../cellbin2')
WEIGHTS_DIR = os.path.join(CELLBIN2_DIR, 'weights')
PATH = os.path.join(WEIGHTS_DIR, 'test_GPU.onnx')
URL = "https://bgipan.genomics.cn/v3.php/download/ln-file?FileId=795370&ShareKey=5iV92x1JzBSwQd77ZAG6&VersionId=608268&UserId=3503&Policy=eyJBSyI6IjdjNmJhYjNkMGZkNWNlZDhjMmNjNzJjNzdjMDc4ZWE3IiwiQWF0IjoxLCJBaWQiOiJKc1FDc2pGM3lyN0tBQ3lUIiwiQ2lkIjoiZjc0YzY3OWQtNjZlZS00NzU5LTg4OWYtZDIzNzNhOWM4NjkyIiwiRXAiOjkwMCwiRGF0ZSI6IlR1ZSwgMDggT2N0IDIwMjQgMDc6MDk6MzUgR01UIn0%3D&Signature=f192cf38b04204f9feb634be2bfcaa5e24a21263"
ONNX_URL = 'https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html'


def download(local_file, file_url):
    f_name = os.path.basename(local_file)
    if not os.path.exists(local_file):
        try:
            r = requests.get(file_url, stream=True)
            total = int(r.headers.get('content-length', 0))
            with open(local_file, 'wb') as fd, tqdm(
                    desc='Downloading {}'.format(f_name), total=total,
                    unit='B', unit_scale=True) as bar:
                for data in r.iter_content(chunk_size=1024):
                    siz = fd.write(data)
                    bar.update(siz)
        except Exception as e:
            print('FAILED! (Download {} from remote {})'.format(f_name, file_url))
            print(e)
            return 1
    else:
        print('{} already exists'.format(f_name))

def check_onnxruntime_env(gpu_id=0):
    print("Starting to check maximum supported CUDA version, actual CUDA version and ONNXRuntime-gpu version...")
    print(f"**Maximum CUDA version supported by GPU (nvidia-smi): {get_nvidia_smi_cuda_version()}")
    print(f"**Actual installed CUDA version: {get_cuda_version()}")
    print(f"**onnxruntime-gpu version: {get_onnxruntime_version()}")
    print("Note: The actual installed CUDA version should not be higher than the maximum supported CUDA version!")
    

    use_list = ort.get_available_providers()
    gpu_available = 'CUDAExecutionProvider' in use_list
    cpu_available = 'CPUExecutionProvider' in use_list

    if not gpu_available:
        if cpu_available:
            print("❌ GPU is not supported, only CPU is available.")
            return {
                "status": "cpu_only",
                "gpu_support": False,
                "reason": "CUDAExecutionProvider is not available. Possible reasons:\n"
                          "1. No NVIDIA GPU detected\n"
                          "2. CUDA is not installed\n"
                          "3. onnxruntime-gpu is not installed (you might have installed the CPU version instead)\n"
                          "4. Version mismatch between onnxruntime-gpu, CUDA, and cuDNN (Confirm the version matching information by accessing the \nURL: {ONNX_URL}）\n"
            }
        else:
            print("❌ Neither GPU nor CPU is supported (abnormal situation)")
            return {
                "status": "no_provider",
                "gpu_support": False,
                "reason": "Neither CUDAExecutionProvider nor CPUExecutionProvider is available. Possible reasons:\n"
                          "1. onnxruntime is not installed correctly\n"
                          "2. Python environment is broken\n"
                          "3. System is missing required dependencies"
            }

    print("✅ GPU is theoretically supported, starting actual verification...")
    if not os.path.exists(PATH):
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        download(PATH, URL)

    try:
        session = ort.InferenceSession(
            PATH,
            providers=[('CUDAExecutionProvider', {'device_id': str(gpu_id)}), 'CPUExecutionProvider']
        )
        active_provider = session.get_providers()[0]
        if active_provider == 'CUDAExecutionProvider':
            print("🎉 Successfully running in GPU mode")
            return {
                "status": "gpu_ok",
                "gpu_support": True,
                "active_provider": active_provider
            }
        elif active_provider == 'CPUExecutionProvider':
            print(f"⚠️ GPU initialization failed, automatically fell back to CPU")
            return {
                "status": "gpu_fallback_cpu",
                "gpu_support": False,
                "active_provider": active_provider,
                "reason": "Failed to initialize CUDAExecutionProvider. Possible reasons:\n"
                          "1. CUDA/cuDNN/onnxruntioiem-gpu version mismatch (Confirm the version matching information by accessing the \nURL: {ONNX_URL}）\n"
                          "2. GPU out of memory\n"
                          "3. Missing CUDA dependencies\n"
                          "4. The selected GPU device is not available or busy (Check the GPU status and try again)"
            }
        else:
            print(f"⚠️ Unexpected fallback to {active_provider}")
            return {
                "status": "unexpected_fallback",
                "gpu_support": False,
                "active_provider": active_provider,
                "reason": "Session did not use CUDAExecutionProvider or CPUExecutionProvider. Possible reasons:\n"
                          "1. Fallback to another provider (e.g. TensorRT, OpenVINO)\n"
                          "2. Provider configuration error\n"
                          f"3. Check the \n{active_provider} \ncontent and consult relevant documentation"
            }
    except Exception as e:
        import traceback
        print(f"❌ Unknown error: {str(e)}")
        traceback.print_exc()
        return {
            "status": "unknown_error",
            "gpu_support": False,
            "error_type": type(e).__name__,
            "error_details": str(e),
            "reason": "Exception occurred during session creation. Possible reasons:\n"
                      "1. CUDA/cuDNN/onnxruntime-gpu version mismatch\n"
                      "2. GPU driver or hardware failure\n"
                      "3. Model file is corrupted or incompatible\n"
                      "4. Python environment issues"
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id to use (default: 0)')
    args = parser.parse_args()
    result = check_onnxruntime_env(gpu_id=args.gpu)
    print("\nDetailed diagnostic information:")
    for k, v in result.items():
        print(f"{k:>20}: {v}")