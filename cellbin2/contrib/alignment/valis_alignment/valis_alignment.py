import argparse
from tracemalloc import start
import numpy as np
from valis import registration
import os
import tempfile
import shutil
import warnings
import logging
import sys
import time
warnings.filterwarnings("ignore")
logging.getLogger("pyvips").setLevel(logging.ERROR)
class NoOutput:
    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self.stdout
        sys.stderr = self.stderr

def get_image_offset(image1_path, image2_path):
    """
    (x_offset, y_offset) 
    """
    
    temp_dir = tempfile.mkdtemp()
    img1_name = "image1.tif"
    img2_name = "image2.tif"
    
    img1_temp = os.path.join(temp_dir, img1_name)
    img2_temp = os.path.join(temp_dir, img2_name)
    
    shutil.copy2(image1_path, img1_temp)
    shutil.copy2(image2_path, img2_temp)
    
    try:
        with NoOutput():
            start = time.time()
            registrar = registration.Valis(temp_dir, temp_dir, do_rigid=True, non_rigid_registrar_cls=None)
            registrar.register()
        
            # 
            slide_dict = registrar.slide_dict
            slides = list(slide_dict.values())
            
            moving_slide = slides[1]  # moving slide
            
            M = moving_slide.M
            
            # 
            x_offset = float(M[0, 2])
            y_offset = float(M[1, 2])
            end = time.time()
        
        return {
            'x_offset_pixels': x_offset,
            'y_offset_pixels': y_offset
        }
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def args_parse():
    usage = """ Usage: %s Alignment of two images using Valis."""
    arg = argparse.ArgumentParser(usage=usage)
    arg.add_argument('-r','--reference',help='reference image path', required=True)
    arg.add_argument('-m', '--move', help='moving image path', required=True)

    return arg.parse_args()


def main():
    args = args_parse()
    ref_img = args.reference
    mov_img = args.move
    # fetch offsets
    result = get_image_offset(ref_img, mov_img)
    
    if result:
        print(f"  X-offset: {result['x_offset_pixels']:.2f} pixels")
        print(f"  Y-offset: {result['y_offset_pixels']:.2f} pixels")


if __name__ == '__main__':
    main()
    
    



