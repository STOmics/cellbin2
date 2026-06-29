# import json
# import os
# from enum import Enum
# from typing import Union, List, Dict
# import requests
# from cellbin2.utils import clog
# from tqdm import tqdm

# from cellbin2.utils.ipr import sPlaceHolder
# CURR_DIR = os.path.dirname(os.path.realpath(__file__))
# CB2_DIR = os.path.dirname(CURR_DIR)
# DEFAULT_WEIGHTS_DIR = os.path.join(CB2_DIR, "weights")



# class DNNModuleName(Enum):
#     cellseg = 1
#     tissueseg = 2
#     clarity = 3
#     points_detect = 4
#     chip_detect = 5


# class DownloadSource(Enum):
#     BGIPAN = "bgipan"
#     GITHUB = "github"


# def download(local_file, file_url, source_name="remote"):
#     f_name = os.path.basename(local_file)
#     if not os.path.exists(local_file):
#         try:
#             r = requests.get(file_url, stream=True, timeout=6000)
#             r.raise_for_status()
            
#             total = int(r.headers.get('content-length', 0))
#             with open(local_file, 'wb') as fd, tqdm(
#                     desc=f'Downloading {f_name} from {source_name}',
#                     total=total,
#                     unit='B',
#                     unit_scale=True) as bar:
#                 for data in r.iter_content(chunk_size=1024):
#                     siz = fd.write(data)
#                     bar.update(siz)
#             clog.info(f'Successfully downloaded {f_name} from {source_name}')
#             return True
#         except Exception as e:
#             clog.error(f'Failed to download {f_name} from {source_name}: {str(e)}')
#             if os.path.exists(local_file):
#                 os.remove(local_file)
#             return False
#     else:
#         clog.info(f'{f_name} already exists')
#         return True

    

# class WeightDownloader(object):
#     def __init__(self, save_dir: str, url_file: str = None):
#         if url_file:
#             self._url_file = url_file
#         else:
#             curr_path = os.path.dirname(os.path.realpath(__file__))
#             self._url_file = os.path.join(curr_path, r'../config/weights_url.json')  
        
#         with open(self._url_file, 'r') as fd:
#             self._WEIGHTS = json.load(fd)
        
#         self._save_dir = save_dir
#         os.makedirs(save_dir, exist_ok=True)
#         clog.info('Weights files will be stored in {}'.format(save_dir))

#     @property
#     def weights_list(self):
#         w = {}
#         for module_name, weights in self._WEIGHTS.items():
#             for weight_name, sources in weights.items():
#                 w[weight_name] = sources
#         return w

#     def _download(self, weight_name: str, sources: Dict):

#         weight_path = os.path.join(self._save_dir, weight_name)
        
#         if os.path.exists(weight_path):
#             clog.info(f'{weight_name} already exists, skip download')
#             return 0
        
#         # BGI
#         bgipan_url = sources.get(DownloadSource.BGIPAN.value)
#         if bgipan_url:
#             clog.info(f'Trying to download {weight_name} from BGIPan...')
#             success = download(weight_path, bgipan_url, DownloadSource.BGIPAN.value)
#             if success:
#                 return 0
#             else:
#                 clog.warning(f'BGIPan download failed for {weight_name}, trying GitHub...')

#                 import time
#                 time.sleep(1)  
        
#         # GitHub
#         github_url = sources.get(DownloadSource.GITHUB.value)
#         if github_url:
#             clog.info(f'Trying to download {weight_name} from GitHub...')
#             success = download(weight_path, github_url, DownloadSource.GITHUB.value)
#             if success:
#                 return 0
#             else:
#                 clog.error(f'GitHub download also failed for {weight_name}')
#         else:
#             clog.error(f'No GitHub fallback URL available for {weight_name}')
        
#         return 1


#     def download_weight_by_names(self, weight_names: List[str]):
#         all_weights = self.weights_list
#         failed_downloads = []
        
#         for weight_name in weight_names:
#             if weight_name not in all_weights:
#                 clog.error(f"{weight_name} not in auto download lists")
#                 failed_downloads.append(weight_name)
#                 continue
            
#             sources = all_weights[weight_name]
#             flag = self._download(weight_name, sources)
#             if flag != 0:
#                 failed_downloads.append(weight_name)
        
#         if failed_downloads:
#             clog.error(f"Failed to download: {failed_downloads}")
#             return 1
#         return 0

#     def download_weights(self, module_name: DNNModuleName, weight_name: str):
#         if module_name.name not in self._WEIGHTS:
#             clog.error(f'Module {module_name.name} not found')
#             return 1
        
#         if weight_name not in self._WEIGHTS[module_name.name]:
#             clog.warning(f'{weight_name} not in [{module_name.name}] module')
#             return 1
        
#         sources = self._WEIGHTS[module_name.name][weight_name]
#         return self._download(weight_name, sources)

#     def download_module_weight(self, module_name: Union[DNNModuleName, list] = None):
#         weights_to_download = []
#         if isinstance(module_name, DNNModuleName):
#             weights_to_download = [module_name]
#         elif isinstance(module_name, list):
#             weights_to_download = module_name
#         elif module_name is None:
#             weights_to_download = list(DNNModuleName.__members__.values())
        
#         failed_modules = []
#         for module in weights_to_download:
#             if module.name not in self._WEIGHTS:
#                 clog.warning(f'Module {module.name} not found in config')
#                 continue
            
#             clog.info(f'Downloading weights for module: {module.name}')
#             module_failed = []
#             for weight_name, sources in self._WEIGHTS[module.name].items():
#                 flag = self._download(weight_name, sources)
#                 if flag != 0:
#                     module_failed.append(weight_name)
            
#             if module_failed:
#                 failed_modules.append((module.name, module_failed))
        
#         if failed_modules:
#             for module_name, failed_weights in failed_modules:
#                 clog.error(f'Module {module_name} failed downloads: {failed_weights}')
#             return 1
#         return 0


# def download_by_names(save_dir: str, weight_names: List[str]):
#     wd = WeightDownloader(save_dir=save_dir)
#     return wd.download_weight_by_names(weight_names=weight_names)


# def download_all_weights(save_dir: str = None):
#     if save_dir is None:
#         save_dir = DEFAULT_WEIGHTS_DIR
#     wd = WeightDownloader(save_dir)
#     return wd.download_module_weight()

# if __name__ == '__main__':
#     save_dir = '/media/Data1/user/dengzhonghan/data/tmp/test_weights_2'
#     download_all_weights()
    
#     # names = [DNNModuleName.cellseg]
#     # wd = WeightDownloader(save_dir)
#     #
#     # # download by module 
#     # wd.download_module_weight(names)
#     #
#     # # download by module/model name
#     # wd.download_weights(names[0], 'cellseg_bcdu_SHDI_221008_tf.onnx')
#     # wd.download_weights(names[0], 'points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx')
#     #
#     # # download by names list 
#     # wd.download_weight_by_names(['chip_detect_yolov5obb_SSDNA_20241001_pytorch.onnx',
#     #                              'tissueseg_yolo_SH_20230131_th.onnx'])
    
import json
import os
from enum import Enum
from typing import Union, List, Dict
from urllib.parse import quote_plus
import requests
from cellbin2.utils import clog
from tqdm import tqdm

from cellbin2.utils.ipr import sPlaceHolder
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
CB2_DIR = os.path.dirname(CURR_DIR)
DEFAULT_WEIGHTS_DIR = os.path.join(CB2_DIR, "weights")
DEFAULT_MODELSCOPE_MODEL_ID = "Aoli9811/cellbin2-weights"
DEFAULT_MODELSCOPE_REVISION = "master"
MODELSCOPE_ENDPOINT = "https://modelscope.cn"

# On ModelScope the weight files are stored under module-named subdirectories
# (e.g. "cellseg/cyto3"). For most weights the filename matches the config key,
# so the path is just "<module>/<weight_name>". A few keys differ from the
# actual filename on ModelScope and are remapped here.
MODELSCOPE_FILENAME_OVERRIDES = {
    "size_cyto2torch_0": "size_cyto2torch_0.npy",
    "cellpose3": "nucleitorch_0",
}



class DNNModuleName(Enum):
    cellseg = 1
    tissueseg = 2
    clarity = 3
    points_detect = 4
    chip_detect = 5


class DownloadSource(Enum):
    MODELSCOPE = "modelscope"
    BGIPAN = "bgipan"
    GITHUB = "github"



def modelscope_file_url(
        file_path: str,
        model_id: str = DEFAULT_MODELSCOPE_MODEL_ID,
        revision: str = DEFAULT_MODELSCOPE_REVISION):
    file_path = quote_plus(file_path)
    revision = quote_plus(revision)
    return (
        f"{MODELSCOPE_ENDPOINT}/api/v1/models/{model_id}/repo"
        f"?Revision={revision}&FilePath={file_path}"
    )



def download(local_file, file_url, source_name="remote"):
    f_name = os.path.basename(local_file)
    if not os.path.exists(local_file):
        try:
            local_dir = os.path.dirname(local_file)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)

            r = requests.get(file_url, stream=True, timeout=6000)
            r.raise_for_status()
            
            total = int(r.headers.get('content-length', 0))
            with open(local_file, 'wb') as fd, tqdm(
                    desc=f'Downloading {f_name} from {source_name}',
                    total=total,
                    unit='B',
                    unit_scale=True) as bar:
                for data in r.iter_content(chunk_size=1024):
                    if data:
                        siz = fd.write(data)
                        bar.update(siz)
            clog.info(f'Successfully downloaded {f_name} from {source_name}')
            return 0
        except Exception as e:
            clog.error(f'Failed to download {f_name} from {source_name}: {str(e)}')
            if os.path.exists(local_file):
                os.remove(local_file)
            return 1
    else:
        clog.info(f'{f_name} already exists')
        return 0


class WeightDownloader(object):
    def __init__(self, save_dir: str, url_file: str = None):
        if url_file:
            self._url_file = url_file
        else:
            curr_path = os.path.dirname(os.path.realpath(__file__))
            self._url_file = os.path.join(curr_path, r'../config/weights_url.json')  
        
        with open(self._url_file, 'r') as fd:
            self._WEIGHTS = json.load(fd)

        # Reverse index: weight_name -> module (ModelScope stores files under
        # module-named subdirectories that mirror the config structure).
        self._weight_to_module = {}
        for module, weights in self._WEIGHTS.items():
            for weight_name in weights:
                self._weight_to_module[weight_name] = module

        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        clog.info('Weights files will be stored in {}'.format(save_dir))

    @property
    def weights_list(self):
        w = {}
        for module_name, weights in self._WEIGHTS.items():
            for weight_name, sources in weights.items():
                w[weight_name] = sources
        return w

    def _modelscope_file_path(self, weight_name: str) -> str:
        """Path of a weight inside the ModelScope repo: <module>/<filename>.

        Falls back to the bare weight_name if the module can't be resolved
        (e.g. a weight not listed in the config), in which case ModelScope will
        404 and the downloader falls through to the next source.
        """
        module = self._weight_to_module.get(weight_name)
        if module is None:
            return weight_name
        fname = MODELSCOPE_FILENAME_OVERRIDES.get(weight_name, weight_name)
        return f"{module}/{fname}"

    def _download(self, weight_name: str, sources: Dict):

        weight_path = os.path.join(self._save_dir, weight_name)

        if os.path.exists(weight_path):
            clog.info(f'{weight_name} already exists, skip download')
            return 0

        if not isinstance(sources, dict):
            sources = {DownloadSource.GITHUB.value: sources}

        modelscope_url = sources.get(DownloadSource.MODELSCOPE.value)
        if not modelscope_url:
            modelscope_url = modelscope_file_url(
                self._modelscope_file_path(weight_name)
            )

        download_sources = [
            (DownloadSource.MODELSCOPE.value, modelscope_url),
            (DownloadSource.BGIPAN.value, sources.get(DownloadSource.BGIPAN.value)),
            (DownloadSource.GITHUB.value, sources.get(DownloadSource.GITHUB.value)),
        ]

        import time
        for source_name, file_url in download_sources:
            if not file_url:
                continue

            clog.info(f'Trying to download {weight_name} from {source_name}...')
            flag = download(weight_path, file_url, source_name)
            if flag == 0:
                return 0

            clog.warning(f'{source_name} download failed for {weight_name}, trying next source...')
            time.sleep(1)

        clog.error(f'All download sources failed for {weight_name}')
        return 1


    def download_weight_by_names(self, weight_names: List[str]):
        all_weights = self.weights_list
        failed_downloads = []
        
        for weight_name in weight_names:
            if weight_name not in all_weights:
                clog.error(f"{weight_name} not in auto download lists")
                failed_downloads.append(weight_name)
                continue
            
            sources = all_weights[weight_name]
            flag = self._download(weight_name, sources)
            if flag != 0:
                failed_downloads.append(weight_name)

        if failed_downloads:
            clog.error(f"Failed to download: {failed_downloads}")
            return 1
        return 0
       

    
    def download_weights(self, module_name: DNNModuleName, weight_name: str):
        if module_name.name not in self._WEIGHTS:
            clog.error(f'Module {module_name.name} not found')
            return 1

        if weight_name not in self._WEIGHTS[module_name.name]:
            clog.warning('{} not in [{}] module'.format(weight_name, module_name.name))
            return 1

        sources = self._WEIGHTS[module_name.name][weight_name]
        return self._download(weight_name, sources)
    
    def download_module_weight(self, module_name: Union[DNNModuleName, list] = None):
        weights_to_download = []
        if isinstance(module_name, DNNModuleName):
            weights_to_download = [module_name]
        elif isinstance(module_name, list):
            weights_to_download = module_name
        elif module_name is None:
            # if parameter not given, download all weights by default
            weights_to_download = list(DNNModuleName.__members__.values())

        failed_modules = []
        for module in weights_to_download:
            if module.name not in self._WEIGHTS:
                clog.warning(f'Module {module.name} not found in config')
                continue

            clog.info(f'Downloading weights for module: {module.name}')
            module_failed = []
            for weight_name in self._WEIGHTS[module.name].keys():
                flag = self.download_weights(module, weight_name)
                if flag != 0:
                    module_failed.append(weight_name)

            if module_failed:
                failed_modules.append((module.name, module_failed))

        if failed_modules:
            for module_name, failed_weights in failed_modules:
                clog.error(f'Module {module_name} failed downloads: {failed_weights}')
            return 1
        return 0


def download_by_names(save_dir: str, weight_names: List[str]):
    wd = WeightDownloader(save_dir=save_dir)
    return wd.download_weight_by_names(weight_names=weight_names)


def download_all_weights(save_dir: str = None):
    if save_dir is None:
        save_dir = DEFAULT_WEIGHTS_DIR
    wd = WeightDownloader(save_dir)
    return wd.download_module_weight()

if __name__ == '__main__':
    save_dir = '/media/Data1/user/dengzhonghan/data/tmp/test_weights_2'
    download_all_weights()
    
    # names = [DNNModuleName.cellseg]
    # wd = WeightDownloader(save_dir)
    #
    # # download by module 
    # wd.download_module_weight(names)
    #
    # # download by module/model name
    # wd.download_weights(names[0], 'cellseg_bcdu_SHDI_221008_tf.onnx')
    # wd.download_weights(names[0], 'points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx')
    #
    # # download by names list 
    # wd.download_weight_by_names(['chip_detect_yolov5obb_SSDNA_20241001_pytorch.onnx',
    #                              'tissueseg_yolo_SH_20230131_th.onnx'])
    