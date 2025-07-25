from cellbin2.dnn import BaseNet


class TensorRTNet(BaseNet):
    def __init__(self, model_path, gpu="-1", num_threads=0):
        super(TensorRTNet, self).__init__()
        self._gpu = int(gpu)

    def _f_init(self):
        pass

    def _f_load_model(self):
        pass

    def f_predict(self, data):
        pass
