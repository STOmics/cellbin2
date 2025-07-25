import os

from cellbin2.utils.common import TechType


class BaseModule:

    @property
    def supported_model(self):
        supported = []
        for i in self.__dict__:
            i = os.path.basename(i)
            if "weights_path" in i:
                s_type_s = i.split("_")[0]
                s_type = TechType[s_type_s]
                supported.append(s_type)
        return supported

    def get_weights_path(self, s_type: TechType):
        try:
            p = getattr(self, f"{s_type.name}_weights_path")
        except AttributeError:
            p = ''
        return p