import os
import json
import pandas as pd
import jsonschema2md

from cellbin2.utils.common import FILES_TO_KEEP, KIT_VERSIONS, KIT_VERSIONS_R
from cellbin2.modules.naming import DumpPipelineFileNaming, DumpImageFileNaming, DumpMatrixFileNaming
from cellbin2.modules.metadata import ProcParam
from cellbin2.utils.config import Config


def write_to_readme():
    import re
    from cellbin2.utils.common import TechType
    sn = 'SN'
    save_dir = "/demo"
    im_type = TechType.DAPI
    im_type2 = TechType.IF
    m_type = TechType.Transcriptomics
    readme_p = "../../README.md"
    with open(readme_p, "r", encoding='utf-8') as f:
        md_cont = f.read()
    pfn = DumpPipelineFileNaming(sn, save_dir=save_dir)
    ifn = DumpImageFileNaming(sn=sn, stain_type=im_type.name, save_dir=save_dir)
    ifn2 = DumpImageFileNaming(sn=sn, stain_type=im_type2.name, save_dir=save_dir)
    mfn = DumpMatrixFileNaming(sn=sn, m_type=m_type.name, save_dir=save_dir)
    all_ = [pfn, ifn, ifn2, mfn]
    table_md = "| File Name | Description |\n| ---- | ---- |\n"
    if_key_word = ["txt", "merged"]
    m_key_word = ["mask"]
    for n in all_:
        attrs = dir(n)
        for name in attrs:
            if name.startswith("__") or name.startswith("_") or not isinstance(n.__class__.__dict__.get(name),
                                                                               property):
                continue
            else:
                value = getattr(n, name)
                doc = getattr(n.__class__, name).__doc__
                value = value.name
                if isinstance(n, DumpImageFileNaming):
                    type_ = n.stain_type
                    if type_ not in [TechType.DAPI.name, TechType.ssDNA, TechType.HE.name] and \
                            any(kw in value for kw in if_key_word):
                        continue
                if isinstance(n, DumpMatrixFileNaming):
                    if any(kw in value for kw in m_key_word):
                        continue
                if isinstance(n, DumpImageFileNaming) or isinstance(n, DumpMatrixFileNaming):
                    if n.__class__.__dict__.get(name) not in FILES_TO_KEEP:
                        continue

                table_md += f"| {value} | {doc} |\n"

    print(table_md)
    updated_markdown_text = re.sub(r'# Outputs\n[\s\S]*?(?=\n#|\Z)', f'# Outputs\n\n{table_md}\n', md_cont)
    readme_p = "../../README.md"
    with open(readme_p, "w") as f:
        f.write(updated_markdown_text)


def input_json_config():
    config_p = "../../cellbin2/config"
    info = []
    for idx, k in enumerate(KIT_VERSIONS + KIT_VERSIONS_R):
        tech, version = k.split("V")
        if k.endswith("R"):
            tech = tech.strip(" ") + " R"
            param_file = os.path.join(config_p, tech + ".json")
        else:
            param_file = os.path.join(config_p, tech.strip(" ") + ".json")
        with open(param_file, 'r') as fd:
            dct = json.load(fd)
        pp = ProcParam(**dct)

        for i, v in pp.image_process.items():
            tmp_info = [tech]
            tmp_info.extend(
                [i, v.chip_detect, v.quality_control, v.tissue_segmentation, v.cell_segmentation,
                 v.channel_align])
            tmp_info.extend([pp.molecular_classify['Transcriptomics'].correct_r, pp.run.qc, pp.run.alignment, pp.run.matrix_extract, pp.run.report, pp.run.annotation])
            if tmp_info in info:
                continue
            info.append(tmp_info)
    df = pd.DataFrame(info, columns=[
        'kit_type', 'stain_type', 'run_chip_detect', 'run_quality_control', "run_tissue_segmentation",
        "run_cell_segmentation",
        'channel_align', 'correct_radius', 'run_qc', 'run_alignment', 'run_matrix_extract', 'run_report',
        'run_annotation'
    ])

    columns = df.columns.tolist()

    markdown_table = "| " + " | ".join(columns) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(columns)) + " |\n"

    for index, row in df.iterrows():
        row_data = [str(x) for x in row.tolist()]
        markdown_table += "| " + " | ".join(row_data) + " |\n"

    with open("config.md", "w") as f:
        f.write(markdown_table)


if __name__ == '__main__':
    input_json_config()
    # write_to_readme()