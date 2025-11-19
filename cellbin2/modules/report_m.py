import json
import os
import re
from cellbin2.modules.report.min_html import operat_html

CURR_PATH = os.path.dirname(os.path.realpath(__file__))
REPORT_MODULE = os.path.join(CURR_PATH, "report")
RESULT_JSON_PARH = os.path.join(REPORT_MODULE, 'js')
JSON_TEMPLATE = os.path.join(RESULT_JSON_PARH, "result_template.js")

RESOLUTION = 500e-6
COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000", "#7E6148",
          "#B09C85", "#3B4992", "#EE0000", "#008B45", "#631879", "#008280", "#BB0021", "#5F559B", "#A20056",
          "#808180", "#1B1919", "#BC3C29", "#0072B5", "#E18727", "#20854E", "#7876B1", "#6F99AD", "#FFDC91",
          "#EE4C97", "#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91", "#AD002A", "#ADB6B6",
          "#1B1919", "#374E55", "#DF8F44", "#00A1D5", "#B24745", "#79AF97", "#6A6599", "#80796B", "#0073C2",
          "#EFC000", "#868686", "#CD534C", "#7AA6DC", "#003C67"]


class Report(object):
    """ report: unlimited format (HTML) """

    def __init__(self, matrics_json, save_path=RESULT_JSON_PARH, json_template=JSON_TEMPLATE):
        """
<<<<<<< Updated upstream
        :param json_template:  json template for report generation. The JSON file template required for generating the report
=======
        :param json_template:  json template for report generation
>>>>>>> Stashed changes
        :param matrics_json:  calculated matrics json file 
        """
        self.json_path = json_template
        with open(json_template, 'r') as file:
            jstring = file.read()
            json_data = self._javascript_tojson(jstring)
            cstring = self._clean_json(json_data)
            self._json = json.loads(cstring)
        with open(matrics_json, 'r') as f:
            self.matrics_data = json.load(f)
        self.save_path = save_path
        self._json["sampleId"] = self.matrics_data["infor"]["sampleId"]
        self._json["pipelineVersion"] = self.matrics_data["infor"]["pipelineVersion"]

    def _javascript_tojson(self, jstring):
        json_data = jstring.split('resultJson = ')[1].strip().strip(';')
        return json_data

    def save_js(self):
        save_file = os.path.join(self.save_path, "result.js")
        js_code = f"const resultJson = {self._json};"
        with open(save_file, "w") as file:
            file.write(js_code)
        return js_code

    def setparam(self):
        self.set_statistic_data()
        self.set_distribution_data()
        self.set_image_list()
        self.set_heatmap_data()
        self.set_cluster_data()
        self.set_image_ipr()

    def _clean_json(self, json_str):
        json_str = re.sub(r'(^|[^:])//\s.*', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)
        json_str = re.sub(r'\}\s*,\s*}', '}}', json_str)
        json_str = re.sub(r'\}\s*,\s*]', '}]', json_str)
        json_str = re.sub(r'\]\s*,\s*}', ']}', json_str)
        json_str = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r'"\1":', json_str)
        json_str = re.sub(r':\s*([a-zA-Z0-9_]+)\s*(?=[,}])', r': "\1"', json_str)
        return json_str

    def set_statistic_data(self):
        def _set_statistic(statistic_result, matrixtype="RNA", cellbin_or_adjusted="CellBin"):
            for item in self._json["cellbin"][matrixtype]["statistics"]["data"]:
                if item["datatype"] == cellbin_or_adjusted:
                    item["cellCount"] = statistic_result["cellCount"]
                    item["meanCellArea"] = statistic_result["meanCellArea"]
                    item["medianCellArea"] = statistic_result["medianCellArea"]
                    item["meanMID"] = statistic_result["meanMID"]
                    item["medianMID"] = statistic_result["medianMID"]
                    item["meanGeneType"] = statistic_result["meanGeneType"]
                    item["medianGeneType"] = statistic_result["medianGeneType"]
                    item["Fraction_cells"] = statistic_result["Fraction_cells_gene"]
                    item["cell_to_tissue"] = statistic_result["cell_to_tissue_MID"]

        if len(self.matrics_data["matrix"]["RNA"]["statistics"]["CellBin"]) > 0:
            _set_statistic(self.matrics_data["matrix"]["RNA"]["statistics"]["CellBin"], matrixtype="gene",
                           cellbin_or_adjusted="CellBin")
        if len(self.matrics_data["matrix"]["RNA"]["statistics"]["Adjusted"]) > 0:
            _set_statistic(self.matrics_data["matrix"]["RNA"]["statistics"]["Adjusted"], matrixtype="gene",
                           cellbin_or_adjusted="Adjusted")
        if len(self.matrics_data["matrix"]["Protein"]["statistics"]["CellBin"]) > 0:
            _set_statistic(self.matrics_data["matrix"]["Protein"]["statistics"]["CellBin"], matrixtype="protein",
                           cellbin_or_adjusted="CellBin")
        if len(self.matrics_data["matrix"]["Protein"]["statistics"]["Adjusted"]) > 0:
            _set_statistic(self.matrics_data["matrix"]["Protein"]["statistics"]["Adjusted"], matrixtype="protein",
                           cellbin_or_adjusted="Adjusted")

    def set_distribution_data(self):
        def _set_quantile(datalist, jsondict):
            jsondict["min"] = datalist[0]
            jsondict["q1"] = datalist[1]
            jsondict["q2"] = datalist[2]
            jsondict["q3"] = datalist[3]
            jsondict["max"] = datalist[4]

        if len(self.matrics_data["matrix"]["RNA"]["distribution"]["CellBin"]) > 0:
            _set_quantile(self.matrics_data["matrix"]["RNA"]["distribution"]["CellBin"]["MID_data"]
                          , self._json["cellbin"]["gene"]["cb_distribution"]["data"][0]["info"])
            _set_quantile(self.matrics_data["matrix"]["RNA"]["distribution"]["CellBin"]["genetype_data"]
                          , self._json["cellbin"]["gene"]["cb_distribution"]["data"][1]["info"])
            if "cellarea_data" in self.matrics_data["matrix"]["RNA"]["distribution"]["CellBin"]:
                _set_quantile(self.matrics_data["matrix"]["RNA"]["distribution"]["CellBin"]["cellarea_data"]
                              , self._json["cellbin"]["gene"]["cb_distribution"]["data"][2]["info"])
            _set_quantile(self.matrics_data["matrix"]["RNA"]["distribution"]["CellBin"]["celldiameter_data"]
                          , self._json["cellbin"]["gene"]["cb_distribution"]["data"][3]["info"])
        if len(self.matrics_data["matrix"]["RNA"]["distribution"]["Adjusted"]) > 0:
            _set_quantile(self.matrics_data["matrix"]["RNA"]["distribution"]["Adjusted"]["MID_data"]
                          , self._json["cellbin"]["gene"]["ad_distribution"]["data"][0]["info"])
            _set_quantile(self.matrics_data["matrix"]["RNA"]["distribution"]["Adjusted"]["genetype_data"]
                          , self._json["cellbin"]["gene"]["ad_distribution"]["data"][1]["info"])
            if "cellarea_data" in self.matrics_data["matrix"]["RNA"]["distribution"]["Adjusted"]:
                _set_quantile(self.matrics_data["matrix"]["RNA"]["distribution"]["Adjusted"]["cellarea_data"]
                              , self._json["cellbin"]["gene"]["ad_distribution"]["data"][2]["info"])
            _set_quantile(self.matrics_data["matrix"]["RNA"]["distribution"]["Adjusted"]["celldiameter_data"]
                          , self._json["cellbin"]["gene"]["ad_distribution"]["data"][3]["info"])
        if len(self.matrics_data["matrix"]["Protein"]["distribution"]["CellBin"]) > 0:
            _set_quantile(self.matrics_data["matrix"]["Protein"]["distribution"]["CellBin"]["MID_data"]
                          , self._json["cellbin"]["protein"]["cb_distribution"]["data"][0]["info"])
            _set_quantile(self.matrics_data["matrix"]["Protein"]["distribution"]["CellBin"]["genetype_data"]
                          , self._json["cellbin"]["protein"]["cb_distribution"]["data"][1]["info"])
            if "cellarea_data" in self.matrics_data["matrix"]["Protein"]["distribution"]["CellBin"]:
                _set_quantile(self.matrics_data["matrix"]["Protein"]["distribution"]["CellBin"]["cellarea_data"]
                              , self._json["cellbin"]["protein"]["cb_distribution"]["data"][2]["info"])
            _set_quantile(self.matrics_data["matrix"]["Protein"]["distribution"]["CellBin"]["celldiameter_data"]
                          , self._json["cellbin"]["protein"]["cb_distribution"]["data"][3]["info"])
        if len(self.matrics_data["matrix"]["Protein"]["distribution"]["Adjusted"]) > 0:
            _set_quantile(self.matrics_data["matrix"]["Protein"]["distribution"]["Adjusted"]["MID_data"]
                          , self._json["cellbin"]["protein"]["ad_distribution"]["data"][0]["info"])
            _set_quantile(self.matrics_data["matrix"]["Protein"]["distribution"]["Adjusted"]["genetype_data"]
                          , self._json["cellbin"]["protein"]["ad_distribution"]["data"][1]["info"])
            if "cellarea_data" in self.matrics_data["matrix"]["Protein"]["distribution"]["Adjusted"]:
                _set_quantile(self.matrics_data["matrix"]["Protein"]["distribution"]["Adjusted"]["cellarea_data"]
                              , self._json["cellbin"]["protein"]["ad_distribution"]["data"][2]["info"])
            _set_quantile(self.matrics_data["matrix"]["Protein"]["distribution"]["Adjusted"]["celldiameter_data"]
                          , self._json["cellbin"]["protein"]["ad_distribution"]["data"][3]["info"])

    def set_heatmap_data(self):
        if len(self.matrics_data["matrix"]["RNA"]["heatmap"]["rawbin"]) > 0:
            self._json["cellbin"]["gene"]["expression"]["heatmapImageObj"]["allBins"] = \
            self.matrics_data["matrix"]["RNA"]["heatmap"]["rawbin"]["img"]
            self._json["cellbin"]["gene"]["expression"]["heatmapImageObj"]["allBinsColorBar"] = \
                self.matrics_data["matrix"]["RNA"]["heatmap"]["rawbin"]["colorbar"]
        if len(self.matrics_data["matrix"]["RNA"]["heatmap"]["tissuebin"]) > 0:
            self._json["cellbin"]["gene"]["expression"]["heatmapImageObj"]["tissueBins"] = \
            self.matrics_data["matrix"]["RNA"]["heatmap"]["tissuebin"]["img"]
            self._json["cellbin"]["gene"]["expression"]["heatmapImageObj"]["tissueBinsColorBar"] = \
                self.matrics_data["matrix"]["RNA"]["heatmap"]["tissuebin"]["colorbar"]

        if len(self.matrics_data["matrix"]["Protein"]["heatmap"]["rawbin"]) > 0:
            self._json["cellbin"]["protein"]["expression"]["heatmapImageObj"]["allBins"] = \
            self.matrics_data["matrix"]["Protein"]["heatmap"]["rawbin"]["img"]
            self._json["cellbin"]["protein"]["expression"]["heatmapImageObj"]["allBinsColorBar"] = \
                self.matrics_data["matrix"]["Protein"]["heatmap"]["rawbin"]["colorbar"]
        if len(self.matrics_data["matrix"]["Protein"]["heatmap"]["tissuebin"]) > 0:
            self._json["cellbin"]["protein"]["expression"]["heatmapImageObj"]["tissueBins"] = \
            self.matrics_data["matrix"]["Protein"]["heatmap"]["tissuebin"]["img"]
            self._json["cellbin"]["protein"]["expression"]["heatmapImageObj"]["tissueBinsColorBar"] = \
                self.matrics_data["matrix"]["Protein"]["heatmap"]["tissuebin"]["colorbar"]

        pass

    def set_image_list(self):  # register images and tissuecut images
        self._json["baseImageList"] = []

        def _set_register_img(imgbase64, staintype, param, value="demo1"):
            tempdic = {
                "label": staintype,
                "value": value,
                "src": {"source": imgbase64,
                        "xref": "x", "yref": "y", "sizing": "stretch", "layer": "below",
                        "x": param["x_star"], "y": param["y_star"],
                        "sizex": param["sizex"], "sizey": param["sizey"]}
            }
            return tempdic

        for strain_type in self.matrics_data['image']["register_img"].keys():
            self._json["baseImageList"].append(
                _set_register_img(self.matrics_data['image']["register_img"][strain_type],
                                  staintype=strain_type, param=self.matrics_data['image']["param"]))

        self._json["tissueseg"]["src"] = self.matrics_data["tissue_img"]
        self._json["image"]["tissuecut"]["heatmapImageObj"]["tissueBins"] = self.matrics_data["tissue_img"]

        pass

    def set_cluster_data(self):
        self._json["cellbin"]["gene"]["clustering"]["data"] = {}

        def _set_cluster_tojson(dict, matrix_type="gene"):
            category_list = dict["category"]
            for i, c in enumerate(sorted(set(category_list), key=int)):
                indices = [index for index, value in enumerate(category_list) if value == c]
                umap_0 = [str(dict["umap"][0][i]) for i in indices]
                umap_1 = [str(dict["umap"][1][i]) for i in indices]
                x = [str(dict["spatial"][0][i]) for i in indices]
                y = [str(dict["spatial"][1][i]) for i in indices]
                temp_dict = {"mode": "markers", "name": f"Cluster {c}", "type": "scattergl", "hovertemplate": " ",
                             "marker": {}}
                temp_dict["x"], temp_dict["y"] = x, y

                temp_dict["marker"]["size"] = 1.56
                temp_dict["marker"]["opacity"] = 1.0
                temp_dict["marker"]["symbol"] = "circle"
                if i < 49:
                    temp_dict["marker"]["color"] = COLORS[i] # if the category is less than 50, use the COLORS list
                self._json["cellbin"][matrix_type]["clustering"]["data"]["spatial"].append(temp_dict)
                temp_dict = {"mode": "markers", "name": f"Cluster {c}", "type": "scattergl", "hovertemplate": " ",
                             "marker": {}}
                temp_dict["x"], temp_dict["y"] = umap_0, umap_1
                temp_dict["marker"]["size"] = 1.56
                temp_dict["marker"]["opacity"] = 1.0
                temp_dict["marker"]["symbol"] = "circle"
                if i < 49:
                    temp_dict["marker"]["color"] = COLORS[i] # if the category is less than 50, use the COLORS list
                self._json["cellbin"][matrix_type]["clustering"]["data"]["umap"].append(temp_dict)

        if len(self.matrics_data["matrix"]["RNA"]["cluster"]) > 0:
            self._json["cellbin"]["gene"]["clustering"]["data"]["spatial"] = []
            self._json["cellbin"]["gene"]["clustering"]["data"]["umap"] = []
            _set_cluster_tojson(self.matrics_data["matrix"]["RNA"]["cluster"], matrix_type="gene")
        if len(self.matrics_data["matrix"]["Protein"]["cluster"]) > 0:
            self._json["cellbin"]["protein"]["clustering"]["data"]["spatial"] = []
            self._json["cellbin"]["protein"]["clustering"]["data"]["umap"] = []
            _set_cluster_tojson(self.matrics_data["matrix"]["Protein"]["cluster"], matrix_type="protein")

    def set_image_ipr(self):
        def _set_data_dict(label, value):
            _temp = {}
            _temp["label"] = label
            _temp["value"] = value
            return _temp

        self._json["image"]["summary"]["data"] = []
        images_number = len(self.matrics_data["image_ipr"].keys()) - 2
        self._json["image"]["summary"]["data"].append(_set_data_dict("The number of Images", images_number))
        self._json["image"]["summary"]["data"].append(_set_data_dict("ImageSizeX (mm)",
                                                                     int(self.matrics_data["image"]["param"][
                                                                             "sizex"]) * RESOLUTION))
        self._json["image"]["summary"]["data"].append(_set_data_dict("ImageSizeY (mm)",
                                                                     int(self.matrics_data["image"]["param"][
                                                                             "sizey"]) * RESOLUTION))
        layers = list(self.matrics_data["image_ipr"].keys())
        layers.remove("ManualState")
        layers.remove("StereoResepSwitch")
        self._json["image"]["image_num"] = len(layers)

        main_stain = set(layers) & set(['HE', 'DAPI', 'ssDNA'])
        if len(main_stain) > 1:
            # choice stain type
            layer_ = max(main_stain, key=lambda x: len(self.matrics_data["image_ipr"].get(x, {}).keys()))
        else:
            layer_ = list(main_stain)[0]
        for num, layer in enumerate(layers):
            if layer != layer_:
                continue

            self._json["image"]["summary"]["data"].append(_set_data_dict(f"Image_{num+1} name", layer))
            self._json["image"]["summary"]["data"].append(_set_data_dict(f"Image_{num+1} channel", self.matrics_data["image_ipr"][layer]["image_info"]["channelcount"]))
            self._json["image"]["summary"]["data"].append(_set_data_dict(f"Image_{num+1} file size (M bits)", self.matrics_data["image_ipr"][layer]["image_info"]["image_size"]))
            ### set image QC infor
            self._json["image"][f"image{num + 1}_qc"]["data"] = []
            self._json["image"][f"image{num + 1}_qc"]["data"].append(_set_data_dict("Image QC version", self.matrics_data["image_ipr"][layer]["QC_info"]["imageqcversion"]))
            pass_or_not = "pass" if int(self.matrics_data["image_ipr"][layer]["QC_info"]["qcpassflag"]) == 1 else "no pass"
            self._json["image"][f"image{num + 1}_qc"]["data"].append(_set_data_dict("QC Pass", pass_or_not))
            trackline_score = self.matrics_data["image_ipr"][layer]["QC_info"]["tracklinescore"]
            self._json["image"][f"image{num + 1}_qc"]["data"].append(_set_data_dict("Trackline Score", trackline_score))
            pass_or_not = "pass" if self.matrics_data["image_ipr"][layer]["QC_info"][
                                        "trackcrossqcpassflag"] == "1" else "no pass"
            self._json["image"][f"image{num + 1}_qc"]["data"].append(_set_data_dict("Trackcross QC Pass", pass_or_not))
            staintype = self.matrics_data["image_ipr"][layer]["QC_info"]["staintype"]
            self._json["image"][f"image{num + 1}_qc"]["data"].append(_set_data_dict("Strain Type", staintype))
            clarityscore = self.matrics_data["image_ipr"][layer]["QC_info"]["clarityscore"]
            self._json["image"][f"image{num + 1}_qc"]["data"].append(_set_data_dict("Clarity Score", clarityscore))

            ### set image trackpoint and chipbox
            self._json["image"][f"image{num + 1}_trackpoint"]["src"] = self.matrics_data["image_ipr"][layer][
                "trackpoint"]
            self._json["image"][f"image{num + 1}_chipbox"]["src"] = self.matrics_data["image_ipr"][layer][
                "chipbox"]

            ### set small trackpoint image
            img_num = len(self._json["image"][f"image{num + 1}_trackpoint"]["small_chip_image"])
            for i in range(img_num):
                #trackpoint small
                self._json["image"][f"image{num + 1}_trackpoint"]["small_chip_image"][f"chip_image{i+1}_src"] = \
                    self.matrics_data["image_ipr"][layer][f"trackpoint_cp_image_{i+1}"]
                self._json["image"][f"image{num + 1}_trackpoint"]["small_tissue_image"][f"tissue_image{i+1}_src"] = \
                    self.matrics_data["image_ipr"][layer][f"trackpoint_tissue_image_{i+1}"]
                #chipbox part
                self._json["image"][f"image{num + 1}_chipbox"]["chipbox_part_image"][f"chipbox_part_image{i+1}_src"] = \
                    self.matrics_data["image_ipr"][layer][f"chipbox_part_image_{i + 1}"]
            # self._json["image"][f"image{num + 1}_trackpoint"]["src"] = self.matrics_data["image_ipr"][layer][
            #     "trackpoint"]



            ### set image register infor
            self._json["image"][f"image{num + 1}_registration"]["data"] = []
            scalex = self.matrics_data["image_ipr"][layer]["register_info"]["scalex"]
            self._json["image"][f"image{num + 1}_registration"]["data"].append(_set_data_dict("ScaleX", scalex))
            scaley = self.matrics_data["image_ipr"][layer]["register_info"]["scaley"]
            self._json["image"][f"image{num + 1}_registration"]["data"].append(_set_data_dict("ScaleY", scaley))
            rotation = self.matrics_data["image_ipr"][layer]["register_info"]["rotation"]
            self._json["image"][f"image{num + 1}_registration"]["data"].append(_set_data_dict("Rotation", rotation))
            flip = self.matrics_data["image_ipr"][layer]["register_info"]["flip"]
            self._json["image"][f"image{num + 1}_registration"]["data"].append(_set_data_dict("Flip", str(flip)))
            offsetx = self.matrics_data["image_ipr"][layer]["register_info"]["offsetx"]
            self._json["image"][f"image{num + 1}_registration"]["data"].append(
                _set_data_dict("Image X Offset", offsetx))
            offsety = self.matrics_data["image_ipr"][layer]["register_info"]["offsety"]
            self._json["image"][f"image{num + 1}_registration"]["data"].append(
                _set_data_dict("Image Y Offset", offsety))
            counterrot = float(self.matrics_data["image_ipr"][layer]["register_info"]["counterrot90"]) * 90
            self._json["image"][f"image{num + 1}_registration"]["data"].append(
                _set_data_dict("Counter Clockwise Rotation", counterrot))

            ### set image matrix
            self._json["image"][f"image{num + 1}_matrix"]["data"] = []
            if "manualscalex" in self.matrics_data["image_ipr"][layer]["register_info"].keys():
                manualx = str(self.matrics_data["image_ipr"][layer]["register_info"]["manualscalex"])
            else:
                manualx = "-"
            self._json["image"][f"image{num + 1}_matrix"]["data"].append(_set_data_dict("Manual scaleX", manualx))
            if "manualscaley" in self.matrics_data["image_ipr"][layer]["register_info"].keys():
                manualy = str(self.matrics_data["image_ipr"][layer]["register_info"]["manualscaley"])
            else:
                manualy = "-"
            self._json["image"][f"image{num + 1}_matrix"]["data"].append(_set_data_dict("Manual scaleY", manualy))
            if "manualrotation" in self.matrics_data["image_ipr"][layer]["register_info"].keys():
                manualrotation = str(self.matrics_data["image_ipr"][layer]["register_info"]["manualrotation"])
            else:
                manualrotation = "-"
            self._json["image"][f"image{num + 1}_matrix"]["data"].append(
                _set_data_dict("Manual Rotation", manualrotation))
            registerscore = str(self.matrics_data["image_ipr"][layer]["register_info"]["registerscore"])
            self._json["image"][f"image{num + 1}_matrix"]["data"].append(
                _set_data_dict("Register Score", registerscore))
            matrixheight = str(self.matrics_data["image_ipr"][layer]["register_info"]["matrixshape"][1])
            matrixwidth = str(self.matrics_data["image_ipr"][layer]["register_info"]["matrixshape"][0])
            self._json["image"][f"image{num + 1}_matrix"]["data"].append(_set_data_dict("Matrix Height", matrixheight))
            self._json["image"][f"image{num + 1}_matrix"]["data"].append(_set_data_dict("Matrix Width", matrixwidth))
            matrix_x_start = str(self.matrics_data["image_ipr"][layer]["register_info"]["xstart"])
            matrix_y_start = str(self.matrics_data["image_ipr"][layer]["register_info"]["ystart"])
            self._json["image"][f"image{num + 1}_matrix"]["data"].append(
                _set_data_dict("Matrix X Start", matrix_x_start))
            self._json["image"][f"image{num + 1}_matrix"]["data"].append(
                _set_data_dict("Matrix Y Start", matrix_y_start))

            ## set cell segmentation infor
            self._json["image"][f"image{num + 1}_cellseg"]["data"] = []
            if "cellseg" in self.matrics_data["image_ipr"][layer].keys():
                area_ratio = self.matrics_data["image_ipr"][layer]["cellseg"]["area_ratio"]
                self._json["image"][f"image{num + 1}_cellseg"]["data"].append(
                    _set_data_dict("Cell Area / Tissue Area", area_ratio))
                area_ratio_cor = self.matrics_data["image_ipr"][layer]["cellseg"]["area_ratio_cor"]
                self._json["image"][f"image{num + 1}_cellseg"]["data"].append(
                    _set_data_dict("Adjusted Cell Area / Tissue Area", area_ratio_cor))
                int_ratio = self.matrics_data["image_ipr"][layer]["cellseg"]["int_ratio"]
                self._json["image"][f"image{num + 1}_cellseg"]["data"].append(
                    _set_data_dict("Cell Intensity / Tissue Intensity", int_ratio))
                
                # Map original 5 cellseg images from images array and overview
                # First map overview
                if "overview" in self.matrics_data["image_ipr"][layer]["cellseg"]:
                    self._json["image"][f"image{num + 1}_cellseg"]["overview"]["src"] = \
                        self.matrics_data["image_ipr"][layer]["cellseg"]["overview"]
                else:
                    self._json["image"][f"image{num + 1}_cellseg"]["overview"]["src"] = ""
                
                # Map original 5 cellseg images from images array to cellseg1-5
                if "images" in self.matrics_data["image_ipr"][layer]["cellseg"]:
                    images_list = self.matrics_data["image_ipr"][layer]["cellseg"]["images"]
                    for i in range(min(5, len(images_list))):  # Only use first 5 images
                        self._json["image"][f"image{num + 1}_cellseg"][f"cellseg{i + 1}"] = \
                            images_list[i]["src"]
                    
                    # Fill remaining slots with empty strings
                    for i in range(len(images_list), 8):  # Fill cellseg6-8 with empty strings
                        self._json["image"][f"image{num + 1}_cellseg"][f"cellseg{i + 1}"] = ""
                else:
                    # If no images array, fill all cellseg slots with empty strings
                    for i in range(1, 9):
                        self._json["image"][f"image{num + 1}_cellseg"][f"cellseg{i}"] = ""
                
                self._json["image"][f"image{num + 1}_cellseg"]["cell_intensity"]["src"] = \
                    self.matrics_data["image_ipr"][layer]["cellseg"]["cell_intensity"]
            if "clarity" in self.matrics_data["image_ipr"][layer].keys():
                self._json["image"][f"image{num + 1}_clarity"]["src"] = \
                    self.matrics_data["image_ipr"][layer]["clarity"]

    def _falsified_data(self, layer):
        """
           when image information cannot be read from ipr, temporarily generate fake names
           when file size data cannot be read from ipr, temporarily generate a fake file size value

        """
        self.matrics_data["image_ipr"][layer]["image_info"]["imagename"] = "SS200000135TL_D1.tif"
        self.matrics_data["image_ipr"][layer]["image_info"]["imagesize"] = "4.12"

        return 0


def creat_report(matric_json, save_path):
    report = Report(matrics_json=matric_json)
    report.setparam()
    report.save_js()
    os.chdir(REPORT_MODULE)
    if os.path.isdir(os.path.join(save_path, "assets")) and not os.path.isdir(os.path.join(save_path, "assets/common")):
        import shutil
        shutil.copytree(r"assets/common", os.path.join(save_path, "assets", "common"))
    operat_html("index.html", os.path.join(save_path, "CellBin_v2.0_report.html"))
    # operat_html("index.html","CellBin_v2.0_report.html")


def main():
    creat_report(
        matric_json=r"F:\01.users\hedongdong\cellbin2_test\report_result\pipline\report\metrics.json",
        save_path=r"F:\01.users\hedongdong\cellbin2_test\report_result\pipline\report")


if __name__ == '__main__':
    main()
