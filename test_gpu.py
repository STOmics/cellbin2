import onnxruntime as ort
from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
import requests
import os
from tqdm import tqdm
# config_file = os.path.join(curr_path, r'../config/cellbin.yaml')

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
CELLBIN2_DIR = os.path.join(CURR_DIR, 'cellbin2')
WEIGHTS_DIR = os.path.join(CELLBIN2_DIR, 'weights')
PATH = os.path.join(CELLBIN2_DIR, 'test_GPU.onnx')
URL = "https://bgipan.genomics.cn/v3.php/download/ln-file?FileId=795370&ShareKey=5iV92x1JzBSwQd77ZAG6&VersionId=608268&UserId=3503&Policy=eyJBSyI6IjdjNmJhYjNkMGZkNWNlZDhjMmNjNzJjNzdjMDc4ZWE3IiwiQWF0IjoxLCJBaWQiOiJKc1FDc2pGM3lyN0tBQ3lUIiwiQ2lkIjoiZjc0YzY3OWQtNjZlZS00NzU5LTg4OWYtZDIzNzNhOWM4NjkyIiwiRXAiOjkwMCwiRGF0ZSI6IlR1ZSwgMDggT2N0IDIwMjQgMDc6MDk6MzUgR01UIn0%3D&Signature=f192cf38b04204f9feb634be2bfcaa5e24a21263"

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

def check_onnxruntime_env():
    """全面检测 ONNX Runtime 运行环境支持情况"""
    # 基础环境检测
    use_list = ort.get_available_providers()
    gpu_available = 'CUDAExecutionProvider' in use_list
    cpu_available = 'CPUExecutionProvider' in use_list

    # 场景1: 完全不支持 GPU
    if not gpu_available:
        if cpu_available:
            print("❌ 环境不支持 GPU，仅支持 CPU")
            return {"status": "cpu_only", "gpu_support": False}
        else:
            print("❌ 环境既不支持 GPU 也不支持 CPU（异常情况）")
            return {"status": "no_provider", "gpu_support": False}


    print("✅ 环境理论支持 GPU，开始实际验证...")
    if not os.path.exists(PATH):
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        download(PATH, URL)

    try:
        # 尝试 GPU 初始化
        session = ort.InferenceSession(
            PATH,
            providers=['CUDAExecutionProvider'],
            provider_options=[{'device_id': '0'}]
        )

        # 获取实际运行的 Provider
        active_provider = session.get_providers()[0]

        # 场景3: 成功运行在 GPU
        if active_provider == 'CUDAExecutionProvider':
            print("🎉 成功运行在 GPU 模式")
            return {
                "status": "gpu_ok",
                "gpu_support": True,
                "active_provider": active_provider
            }

        # 场景4: 自动回退到 CPU
        elif active_provider == 'CPUExecutionProvider':
            print(f"⚠️ GPU 初始化失败，自动回退到 CPU")
            return {
                "status": "gpu_fallback_cpu",
                "gpu_support": False,
                "active_provider": active_provider,
                "reason": "Possible reasons:\n"
                          "1. CUDA/cuDNN version mismatch\n"
                          "2. GPU out of memory\n"
                          "3. Missing CUDA dependencies"
            }

        # 场景5: 其他异常回退（如 TensorRT）
        else:
            print(f"⚠️ 意外回退到 {active_provider}")
            return {
                "status": "unexpected_fallback",
                "gpu_support": False,
                "active_provider": active_provider
            }

    except RuntimeException as e:
        # 场景6: GPU 初始化抛出明确异常
        print(f"❌ GPU 初始化失败: {str(e)}")
        return {
            "status": "gpu_init_failed",
            "gpu_support": False,
            "error_type": "RuntimeException",
            "error_details": str(e),
            "solution": "Check:\n"
                        "1. CUDA/cuDNN installation\n"
                        "2. onnxruntime-gpu package version\n"
                        "3. GPU driver status"
        }

    except Exception as e:
        # 场景7: 其他未知异常
        print(f"❌ 未知错误: {str(e)}")
        return {
            "status": "unknown_error",
            "gpu_support": False,
            "error_type": type(e).__name__,
            "error_details": str(e)
        }


# 使用示例
if __name__ == "__main__":
    result = check_onnxruntime_env()
    print("\n详细诊断信息:")
    for k, v in result.items():
        print(f"{k:>20}: {v}")