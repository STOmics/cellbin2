import onnxruntime
from os import path
from cellbin2.dnn import BaseNet
from cellbin2.utils import clog


class OnnxNet(BaseNet):
    def __init__(self, model_path, gpu="-1", num_threads=0):
        super(OnnxNet, self).__init__()
        self._providers = ['CPUExecutionProvider']
        self._providers_id = [{'device_id': -1}]
        self._model = None
        self._gpu = int(gpu)
        self._model_path = model_path
        self._input_name = 'input_1'
        self._output_name = None
        self._num_threads = num_threads
        self._input_shape = (0, 0, 0)
        self._f_init()

    def _f_init(self):
        if self._gpu > -1:
            self._providers = ['CUDAExecutionProvider']
            self._providers_id = [{'device_id': self._gpu}]
        self._f_load_model()

    def _f_load_model(self):
        if path.exists(self._model_path):
            clog.info(f"loading weight from {self._model_path}")
            sessionOptions = onnxruntime.SessionOptions()
            try:
                if (self._gpu < 0) and (self._num_threads > 0):
                    sessionOptions.intra_op_num_threads = self._num_threads
                self._model = onnxruntime.InferenceSession(self._model_path, providers=self._providers,
                                                           provider_options=self._providers_id,
                                                           sess_options=sessionOptions)
                if self._gpu < 0:
                    clog.info(f"onnx work on cpu,threads {self._num_threads}")
                else:
                    clog.info(f"onnx work on gpu {self._gpu}")
            except:
                if self._num_threads > 0:
                    sessionOptions.intra_op_num_threads = self._num_threads
                self._model = onnxruntime.InferenceSession(self._model_path, providers=['CPUExecutionProvider'],
                                                           provider_options=[{'device_id': -1}],
                                                           sess_options=sessionOptions)
                clog.info(f"onnx work on cpu,threads {self._num_threads}")
            self._input_name = self._model.get_inputs()[0].name
            self._input_shape = tuple(self._model.get_inputs()[0].shape[1:])
            self._output_shape = tuple(self._model.get_outputs()[0].shape)
        else:
            raise Exception(f"Weight path '{self._model_path}' does not exist")

    def f_predict(self, data):
        pred = self._model.run(self._output_name, {self._input_name: data})
        return pred

    def f_get_input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape
