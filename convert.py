import argparse
import numpy as np
from scipy import ndimage
import os
import SimpleITK as sitk
import cv2

import xml.etree.ElementTree as ET

from contour import contour2nii

def get_argparse():
    parser = argparse.ArgumentParser(description='Argparser')

    parser.add_argument('--data-root', type=str, default= './dataset')
    parser.add_argument('--save-root', type=str, default= './data/VISTA')
    parser.add_argument('--label-root', type=str, default= './data/ROI')

    args = parser.parse_args()

    return args

def maybe_mkdir(dir):
    try:
        os.mkdir(dir)
    except:
        print(f"dir already exists!")
        pass

def load_dicom(dcm_dir):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(dcm_dir))

    return sitk.GetArrayFromImage(reader.Execute())

def dcm2nii(data_root, save_root):
    dirlen = len(os.listdir(data_root))
    print("datalen:", dirlen)
    
    for case_i in sorted(os.listdir(data_root)):
        case_dir = os.path.join(data_root, case_i)

        dcm_img = load_dicom(case_dir)
        print(f"image shape {dcm_img.shape}")

        save_nii(case_dir, f'{save_root}/{case_i}')

def save_nii(dcm_dir, save_dir):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(dcm_dir))
    image3D = reader.Execute()
    
    image_array = sitk.GetArrayViewFromImage(image3D)

    shape = image_array.shape
    mid = shape[2]//2
    
    outputL = sitk.GetImageFromArray(image_array)[mid:]
    outputR = sitk.GetImageFromArray(image_array)[:mid]

    sitk.WriteImage(outputL, f'{save_dir}L_0000.nii.gz')
    sitk.WriteImage(outputR, f'{save_dir}R_0000.nii.gz')


if __name__ == "__main__":
    args = get_argparse()

    maybe_mkdir(args.save_root)
    dcm2nii(args.data_root, args.save_root)
    contour2nii(args.data_root, args.label_root)