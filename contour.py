import argparse
import numpy as np
from scipy import ndimage
import os
import SimpleITK as sitk
import cv2
import json
import xml.etree.ElementTree as ET

def get_contour(qvsroot, slice_id, cont_type, height, width):
    qvas_img = qvsroot.findall('QVAS_Image')
    conts = qvas_img[slice_id].findall('QVAS_Contour')
    pts = None
    for cont_id, cont in enumerate(conts):
        if cont.find('ContourType').text == cont_type:
            pts = cont.find('Contour_Point').findall('Point')
            break
    if pts is not None:
        contours = []
        for p in pts:
            contx = float(p.get('x')) / 512 * width
            conty = float(p.get('y')) / 512 * height
            # if current pt is different from last pt, add to contours
            if len(contours) == 0 or contours[-1][0] != contx or contours[-1][1] != conty:
                contours.append([contx, conty])
        return np.array(contours)
    return None

def fill_image_inv(img):
    h, w = img.shape[:2]
    img_border = np.zeros((h + 2, w + 2), np.uint8)
    img_border[1:-1, 1:-1] = img

    floodfilled = img_border.copy()
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(floodfilled, mask, (0, 0), 1)
    floodfill_inv = cv2.bitwise_not(floodfilled)

    floodfill_inv = floodfill_inv[1:-1, 1:-1]
    floodfill_inv[floodfill_inv == 254] = 0

    return floodfill_inv

def get_fullcontour(pt, shape):
    bg = np.zeros(shape, np.uint8)

    pt = np.array(pt, np.int32)
    pts = pt.reshape((-1,1,2))
    line1 = cv2.polylines(bg.copy(), pts, True, (1,1), 1)
    line2 = cv2.polylines(bg.copy(), pts, True, (1,1), 2)
    poly_inv = fill_image_inv(line1 + line2)
    contour, _ = cv2.findContours(poly_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return np.squeeze(contour), np.array(line1)

def get_roi(lumen, wall, shape):   
    lumenbg = np.zeros(shape, np.uint8)
    wallbg  = np.zeros(shape, np.uint8)

    lumencontour, line1 = get_fullcontour(lumen, shape)
    np.set_printoptions(threshold=256,linewidth=np.inf)

    lumenPoly = cv2.fillConvexPoly(lumenbg, lumencontour, (1,1))

    wallcontour, _ = get_fullcontour(wall, shape)
    wallPoly = cv2.fillConvexPoly(wallbg, wallcontour, (1,1))

    roi = wallPoly + lumenPoly - line1

    return roi


def get_contourarr(qvsroot, slice_id, cont_type, height, width):
    qvas_img = qvsroot.findall('QVAS_Image')
    conts = qvas_img[slice_id].findall('QVAS_Contour')
    pts = None
    
    contours = np.zeros((height, width))
    
    for cont_id, cont in enumerate(conts):
        if cont.find('ContourType').text == cont_type:
            pts = cont.find('Contour_Point').findall('Point')
            break
    if pts is not None:
        for p in pts:
            contx = int(float(p.get('x')) / 512 * width )
            conty = int(float(p.get('y')) / 512 * height)
            
            contours[conty  , contx  ] = 1
        
    return contours

def get_loc_prop(qvj_root, bif_slice):
    loc_label = {}
    
    loc_prop = qvj_root.find('Location_Property')
    for loc in loc_prop.iter('Location'):
        loc_ind = int(loc.get('Index')) + bif_slice
        image_quality = int(loc.find('IQ').text)
        # only slices with Image Quality (IQ) > 1 were labeled 
        # AHAStatus: 1: Normal; > 1 : Atherosclerotic
        AHA_status = float(loc.find('AHAStatus').text)
        if image_quality>1 and AHA_status == 1:
            loc_label[loc_ind] = 0
        elif image_quality>1 and AHA_status >1:
            loc_label[loc_ind] = 1
    
    return loc_label

def get_bir_slice(qvjroot):
    if qvjroot.find('QVAS_System_Info').find('BifurcationLocation'):
        bif_slice = int(qvjroot.find('QVAS_System_Info').find('BifurcationLocation').find('BifurcationImageIndex').get('ImageIndex'))
        return bif_slice
    else:
        return -1


def load_dicom(dcm_dir):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(dcm_dir))
    return sitk.GetArrayFromImage(reader.Execute())

def get_qvs_fname(qvj_path):
    qvs_element = ET.parse(qvj_path).getroot().find('QVAS_Loaded_Series_List').find('QVASSeriesFileName')
    return qvs_element.text


def list_contour_slices(qvs_root):
    """
    :param qvs_root: xml root
    :return: slices with annotations
    """
    avail_slices = []
    image_elements = qvs_root.findall('QVAS_Image')
    for slice_id, element in enumerate(image_elements):
        conts = element.findall('QVAS_Contour')
        if len(conts) > 0:
            avail_slices.append(slice_id)
    return avail_slices


def save_contour(dcm_dir, array, case_i, save_dir):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(dcm_dir))

    shape = array.shape
    mid = shape[2]//2
    
    savedImg = sitk.GetImageFromArray(array)[mid:]
    savedImg = sitk.GetImageFromArray(array)[:mid]

    sitk.WriteImage(savedImg, f'{save_dir}/{case_i}L.nii.gz')
    sitk.WriteImage(savedImg, f'{save_dir}/{case_i}R.nii.gz')

def contour2nii(data_root, save_root):
    arteries = ['L', 'R']
    dirlen = len(os.listdir(data_root))
    print(dirlen)
    
    slice_label = {}
    
    for case_i in sorted(os.listdir(data_root)):
        case_dir = os.path.join(data_root, case_i)
            
        # read dicom
        dcm_img = load_dicom(case_dir)
        d, h, w = dcm_img.shape
        print(f"image shape {dcm_img.shape}")
        
        lumen = np.zeros(dcm_img.shape)
        wall  = np.zeros(dcm_img.shape)
        roi = np.zeros(dcm_img.shape)

        # save contour
        if os.path.isdir(case_dir):
            for art_i in arteries:
                # qvj file record information
                qvj_file = os.path.join(case_dir, case_i + art_i + '.QVJ')
                if os.path.exists(qvj_file):
                    print(f"qvj_file: {qvj_file}")
                    qvs_file = os.path.join(case_dir, get_qvs_fname(qvj_file))
                    qvs_root = ET.parse(qvs_file).getroot()
                    
                    annotated_slices = list_contour_slices(qvs_root)
                    print(f"annotated_slices: {annotated_slices}")

                    if case_i not in slice_label:
                        slice_label[case_i] = {}

                    for anno_id in annotated_slices:
                        lumen[anno_id,:,:] = get_contourarr(qvs_root, anno_id, 'Lumen'     , height=h, width=w)
                        wall[ anno_id,:,:] = get_contourarr(qvs_root, anno_id, 'Outer Wall', height=h, width=w)

                        roi[  anno_id,:,:] = get_roi(get_contour(qvs_root, anno_id, 'Lumen'     , height=h, width=w), 
                                                     get_contour(qvs_root, anno_id, 'Outer Wall', height=h, width=w), dcm_img.shape[1:3])
                        
                    qvj_root = ET.parse(qvj_file).getroot()
                    bif_slice = get_bir_slice(qvj_root)
                    slice_label[case_i][art_i] = get_loc_prop(qvj_root, bif_slice)
        
        # save contour to nii
        save_contour(case_dir, roi , case_i, save_root)

    with open(f'./data/label.json', 'w') as fp:
        json.dump(slice_label, fp)