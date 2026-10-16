import ctypes
import os, sys
import subprocess
from optparse import OptionParser
import time
import warnings
warnings.filterwarnings("ignore")


def newtissue(bin1tif, outpath, modpath, log):
    from tissuecut.tissuecut_api import ClBcduRNAOnnx
    import tifffile

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    bc = ClBcduRNAOnnx(modpath)
    img = tifffile.imread(bin1tif)
    pred = bc.f_predict(img)
    bigtiff = pred.size >= 2**32  # True if image size exceeds 4GB
    tile = [512, 512] if bigtiff else None  # Use tiling for BigTIFF

    try:
        tifffile.imwrite(
            outpath,
            pred,
            compression="zlib",
            compressionargs={"level": 8},
            bigtiff=bigtiff,
            tile=tile
        )
    except Exception as e:
        log.writeErrlog(2,
                        bytes(f"IO ERROR: Failed to save RNA tissue cut tif, the reason for failure is: {e}", 'utf-8'))

    # Delete bin1_img.tif
    os.remove(bin1tif)


def tslib(develop, thcnt, gefpath, outdir, stainfile, dnbfile, binlist, label):
    t0 = time.time()
    programDir, filename = os.path.split(os.path.abspath(__file__))
    tsLibPath = os.path.join(programDir, 'libts.so')

    tslib = ctypes.cdll.LoadLibrary(tsLibPath)
    tslib.initLog()
    modpath = os.path.join(programDir, 'weight', 'tissueseg_bcdu_rna_220909_tf.onnx')

    if gefpath == None:
        tslib.writeErrlog(3, bytes("MISSING PARAMETER: -i", 'utf-8'))
        sys.exit(1)
    if outdir == None:
        tslib.writeErrlog(3, bytes("MISSING PARAMETER: -o", 'utf-8'))
        sys.exit(1)
    t1 = time.time()
    tslib.writeErrlog(0, bytes("Loading libts.so elapsed time(s):{:.0f}".format(t1 - t0), 'utf-8'))

    os.makedirs(outdir, exist_ok=True)
    tslib.readGef(bytes(gefpath, 'utf-8'), bytes(outdir, 'utf-8'))
    if stainfile is None : #无图
        if not os.path.exists(modpath):
            tslib.writeErrlog(1, bytes("CANNOT ACCESS: Model file - %s" % (modpath,), 'utf-8'))
            sys.exit(1)
        t11 = time.time()
        tslib.createBinImg()
        t2 = time.time()
        tslib.writeErrlog(0, bytes("createBinImg time used: {:.3f}".format(t2 - t11), 'utf-8'))
        bin1tif = os.path.join(outdir, "bin1_img.tif")
        stainfile = os.path.join(outdir, "bin1_img_tissue_cut.tif")
        newtissue(bin1tif, stainfile, modpath, tslib)
        t3 = time.time()
        tslib.writeErrlog(0, bytes("create bin1_img_tissue_cut.tif time used: {:.3f}".format(t3 - t2), 'utf-8'))
    else: #有图
        if not os.path.exists(stainfile):
            tslib.writeErrlog(1, bytes("CANNOT ACCESS: -s {}".format(stainfile), 'utf-8'))
            sys.exit(1)

    ret = tslib.dotissuecut(develop, thcnt, bytes(stainfile, 'utf-8'), bytes(dnbfile, 'utf-8'), 
                            bytes(binlist, 'utf-8'), bytes(label, 'utf-8'))
    t5 = time.time()
    tslib.writeErrlog(0, bytes("Tissuecut Done! Total time used: {:.3f}".format(t5 - t0), 'utf-8'))

def main():
    parser = OptionParser(version="%prog v3.10.0")
    parser.add_option("-i", "--input", dest="infile", help="The input gene expression file. ")
    parser.add_option("-o", "--outpath", dest="out_path", help="The output path. ")
    parser.add_option("-t","--thread", dest="thread", default="8", help="set the number of threads. [optional]")
    parser.add_option("--dnbfile", dest='dnbfile', default="", help="The barcode reads count file. [optional]")
    parser.add_option("-s", "--stain", dest='stainfile', help="The tissue segmentation mask file. [optional]")
    parser.add_option("-d", "--develop", dest="develop", default=False, action="store_true", help="Sets whether to output statistic files. [optional]")
    parser.add_option("-b", "--binlist", dest="binlist", default="1,20,50,100,150,200", help="set stat binsize list. [optional]")
    parser.add_option("-l", "--label", dest="label", default="", help="set out name label. [optional]")
    
    opts, args = parser.parse_args()
   
    newtif = os.path.join(opts.out_path, "new_tissue_cut_img.tif")
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./resizetiff.py"))
    command = [
        sys.executable,
        script_path,
        "-i", opts.stainfile,
        "-o", newtif
    ]
    subprocess.run(command, check=True)

    tslib(opts.develop, int(opts.thread,10), opts.infile, opts.out_path, newtif, opts.dnbfile, opts.binlist, opts.label)


if __name__ == '__main__':
    main()
