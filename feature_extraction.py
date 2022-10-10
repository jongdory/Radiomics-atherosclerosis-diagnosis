import radiomics
from radiomics import featureextractor, firstorder, glcm, imageoperations, shape, glrlm, glszm

import SimpleITK as sitk
import os
import numpy
import scipy.io as sio
import time
import numpy as np
import pandas as pd
import json
import re

import logging
# ... or set level for specific class
logger = logging.getLogger("radiomics.glcm")
logger.setLevel(logging.ERROR)

from utility import *

if __name__ == '__main__':

    arteries   = ['L', 'R']
    with open("./data/label.json", "r") as json_file:
        labels = json.load(json_file)

    label  = []
    p_info = []

    ShapeFeatureStorage  = []
    HistFeaturesStorage  = []
    GLCMFeaturesStorage  = []
    GLSZMFeaturesStorage = []
    GLRLMFeaturesStorage = []
    NGTDMFeaturesStorage = []
    GLDMFeaturesStorage  = []

    precase_i = None
    for case_i in sorted(os.listdir(f'./data/ROI')):
        case_i = re.sub(r'[^0-9]', '', case_i)
        if precase_i is not None and precase_i == case_i: continue
        precase_i = case_i

        print(" ")
        print("" + str(case_i) + " Processing Start...")

        glcmFeaturesExtractor = featureextractor.RadiomicsFeatureExtractor(binwidth=128, verbose=True, interpolator=None)
        glcmFeaturesExtractor.disableAllFeatures()
        glcmFeaturesExtractor.enableFeatureClassByName('glcm')

        glszmFeaturesExtractor = featureextractor.RadiomicsFeatureExtractor(binwidth=64, verbose=True, interpolator=None)
        glszmFeaturesExtractor.disableAllFeatures()
        glszmFeaturesExtractor.enableFeatureClassByName('glszm')

        glrlmFeaturesExtractor = featureextractor.RadiomicsFeatureExtractor(binwidth=128, verbose=True, interpolator=None)
        glrlmFeaturesExtractor.disableAllFeatures()
        glrlmFeaturesExtractor.enableFeatureClassByName('glrlm')

        ngtdmFeaturesExtractor = featureextractor.RadiomicsFeatureExtractor(binwidth=128, verbose=True, interpolator=None)
        ngtdmFeaturesExtractor.disableAllFeatures()
        ngtdmFeaturesExtractor.enableFeatureClassByName('ngtdm')

        gldmFeaturesExtractor = featureextractor.RadiomicsFeatureExtractor(binwidth=128, verbose=True, interpolator=None)
        gldmFeaturesExtractor.disableAllFeatures()
        gldmFeaturesExtractor.enableFeatureClassByName('gldm')

        for art_i in arteries:
            # Load images
            VesselimageName = f'./data/VISTA/{case_i}{art_i}_0000.nii.gz'
            Vesselimage = sitk.ReadImage(VesselimageName)
            Vessel_arr  = sitk.GetArrayFromImage(Vesselimage)
            Vesselimage = sitk.GetImageFromArray(Vessel_arr)
            
            # Type Casting
            ROIName = f'./data/ROI/{case_i}{art_i}.nii.gz'
            ROI     = sitk.ReadImage(ROIName)
            roi_arr = sitk.GetArrayFromImage(ROI)
            ROI     = sitk.GetImageFromArray(roi_arr)

            ROI.SetSpacing(Vesselimage.GetSpacing())
            ROI.SetOrigin(Vesselimage.GetOrigin())

            Lumen_ROI  = (ROI == 2)
            Outer_ROI  = (ROI == 1)
            Vessel_ROI = (ROI == 1) + (ROI == 2)

            try:
                anno_slice = labels[case_i][art_i]
            except:
                continue

            for anno_i in anno_slice:
                anno_i = int(anno_i)
                if anno_i not in np.unique(np.where(roi_arr==1)[0]):
                    continue
                
                try:
                    label.append(labels[case_i][art_i][str(anno_i)])
                    p_info.append(f"{case_i}_{art_i}_{anno_i}")
                    print("Calculating feature case:", case_i, art_i, anno_i, "...")
                except:
                    print("error case:", case_i, art_i, anno_i)
                    continue

                try:
                    glcmFeatures_ow  = glcmFeaturesExtractor.execute(Vesselimage[:,:,anno_i],  Outer_ROI[:,:,anno_i])
                    glcmFeatures_ve  = glcmFeaturesExtractor.execute(Vesselimage[:,:,anno_i], Vessel_ROI[:,:,anno_i])

                    glszmFeatures_ow = glszmFeaturesExtractor.execute(Vesselimage[:,:,anno_i],  Outer_ROI[:,:,anno_i])
                    glszmFeatures_ve = glszmFeaturesExtractor.execute(Vesselimage[:,:,anno_i], Vessel_ROI[:,:,anno_i])

                    glrlmFeatures_ow = glrlmFeaturesExtractor.execute(Vesselimage[:,:,anno_i],  Outer_ROI[:,:,anno_i])
                    glrlmFeatures_ve = glrlmFeaturesExtractor.execute(Vesselimage[:,:,anno_i], Vessel_ROI[:,:,anno_i])

                    ngtdmFeatures_ow = ngtdmFeaturesExtractor.execute(Vesselimage[:,:,anno_i],  Outer_ROI[:,:,anno_i])
                    ngtdmFeatures_ve = ngtdmFeaturesExtractor.execute(Vesselimage[:,:,anno_i], Vessel_ROI[:,:,anno_i])

                    gldmFeatures_ow  = gldmFeaturesExtractor.execute(Vesselimage[:,:,anno_i],  Outer_ROI[:,:,anno_i])
                    gldmFeatures_ve  = gldmFeaturesExtractor.execute(Vesselimage[:,:,anno_i], Vessel_ROI[:,:,anno_i])

                    shapeFeatures    = np.array(list(ShapeFeaturesExtractor(Vesselimage[:,:,anno_i],  Outer_ROI[:,:,anno_i] , 'OuterWall').values()) + \
                                                list(ShapeFeaturesExtractor(Vesselimage[:,:,anno_i], Vessel_ROI[:,:,anno_i] , 'Vessel').values())).squeeze()

                    histoFeatures    = np.array(list(firstOrderFeaturesExtractor(Vesselimage[:,:,anno_i],  Outer_ROI[:,:,anno_i] , 'OuterWall').values()) + \
                                                list(firstOrderFeaturesExtractor(Vesselimage[:,:,anno_i], Vessel_ROI[:,:,anno_i] , 'Vessel').values())).squeeze()
                except:
                    anno_i = anno_i - 5
                    continue


                # GLCM Based Features, LoG OFF
                GLCMFeaturesStorage.append(np.array([glcmFeatures_ow[x] for x in list(filter(lambda k: k.startswith("original_"), glcmFeatures_ow))] + \
                                                        [glcmFeatures_ve[x] for x in list(filter(lambda k: k.startswith("original_"), glcmFeatures_ve))]).squeeze())

                GLSZMFeaturesStorage.append(np.array([glszmFeatures_ow[x] for x in list(filter(lambda k: k.startswith("original_"), glszmFeatures_ow))] + \
                                                        [glszmFeatures_ve[x] for x in list(filter(lambda k: k.startswith("original_"), glszmFeatures_ve))]).squeeze())

                # GLRLM Based Features, LoG OFF
                GLRLMFeaturesStorage.append(np.array([glrlmFeatures_ow[x] for x in list(filter(lambda k: k.startswith("original_"), glrlmFeatures_ow))] + \
                                                        [glrlmFeatures_ve[x] for x in list(filter(lambda k: k.startswith("original_"), glrlmFeatures_ve))]).squeeze())

                # NGTDM Based Features, LoG OFF
                NGTDMFeaturesStorage.append(np.array([ngtdmFeatures_ow[x] for x in list(filter(lambda k: k.startswith("original_"), ngtdmFeatures_ow))] + \
                                                        [ngtdmFeatures_ve[x] for x in list(filter(lambda k: k.startswith("original_"), ngtdmFeatures_ve))]).squeeze())

                # GLDM Based Features, LoG OFF
                GLDMFeaturesStorage.append(np.array([gldmFeatures_ow[x] for x in list(filter(lambda k: k.startswith("original_"), gldmFeatures_ow))] + \
                                                    [gldmFeatures_ve[x] for x in list(filter(lambda k: k.startswith("original_"), gldmFeatures_ve))]).squeeze())

                # Shape Based Features, LoG off
                ShapeFeatureStorage.append(shapeFeatures)

                # Histogram Based Features, LoG off
                HistFeaturesStorage.append(histoFeatures)


    # Set Feature name

    ShapeFeaturesNames_O = list(ShapeFeaturesExtractor(        Vesselimage[:,:,anno_i], Outer_ROI[:,:,anno_i]  ,'OuterWall').keys())
    ShapeFeaturesNames_V = list(ShapeFeaturesExtractor(        Vesselimage[:,:,anno_i], Vessel_ROI[:,:,anno_i] ,'Vessel'   ).keys())

    HistFeaturesNames_O  = list(firstOrderFeaturesExtractor(   Vesselimage[:,:,anno_i], Outer_ROI[:,:,anno_i]  ,'OuterWall').keys())
    HistFeaturesNames_V  = list(firstOrderFeaturesExtractor(   Vesselimage[:,:,anno_i], Vessel_ROI[:,:,anno_i] ,'Vessel'   ).keys())

    GLCMFeaturesNames_O = list(filter(lambda k: k.startswith("original_"), glcmFeaturesExtractor.execute(Vesselimage[:,:,anno_i], Outer_ROI[:,:,anno_i]).keys()))
    GLCMFeaturesNames_O = [s.replace('original_glcm', 'GLCM') + "_OuterWall" for s in GLCMFeaturesNames_O]
    GLCMFeaturesNames_V = list(filter(lambda k: k.startswith("original_"), glcmFeaturesExtractor.execute(Vesselimage[:,:,anno_i],Vessel_ROI[:,:,anno_i]).keys()))
    GLCMFeaturesNames_V = [s.replace('original_glcm', 'GLCM') + "_Vessel"    for s in GLCMFeaturesNames_V]

    GLSZMFeaturesNames_O = list(filter(lambda k: k.startswith("original_"), glszmFeaturesExtractor.execute(Vesselimage[:,:,anno_i], Outer_ROI[:,:,anno_i]).keys()))
    GLSZMFeaturesNames_O = [s.replace('original_glszm', 'GLSZM') + "_OuterWall" for s in GLSZMFeaturesNames_O]
    GLSZMFeaturesNames_V = list(filter(lambda k: k.startswith("original_"), glszmFeaturesExtractor.execute(Vesselimage[:,:,anno_i],Vessel_ROI[:,:,anno_i]).keys()))
    GLSZMFeaturesNames_V = [s.replace('original_glszm', 'GLSZM') + "_Vessel"    for s in GLSZMFeaturesNames_V]

    GLRLMFeaturesNames_O = list(filter(lambda k: k.startswith("original_"), glrlmFeaturesExtractor.execute(Vesselimage[:,:,anno_i], Outer_ROI[:,:,anno_i]).keys()))
    GLRLMFeaturesNames_O = [s.replace('original_glrlm', 'GLRLM') + "_OuterWall" for s in GLRLMFeaturesNames_O]
    GLRLMFeaturesNames_V = list(filter(lambda k: k.startswith("original_"), glrlmFeaturesExtractor.execute(Vesselimage[:,:,anno_i],Vessel_ROI[:,:,anno_i]).keys()))
    GLRLMFeaturesNames_V = [s.replace('original_glrlm', 'GLRLM') + "_Vessel"    for s in GLRLMFeaturesNames_V]

    NGTDMFeaturesNames_O = list(filter(lambda k: k.startswith("original_"), ngtdmFeaturesExtractor.execute(Vesselimage[:,:,anno_i], Outer_ROI[:,:,anno_i]).keys()))
    NGTDMFeaturesNames_O = [s.replace('original_ngtdm', 'NGTDM') + "_OuterWall" for s in NGTDMFeaturesNames_O]
    NGTDMFeaturesNames_V = list(filter(lambda k: k.startswith("original_"), ngtdmFeaturesExtractor.execute(Vesselimage[:,:,anno_i],Vessel_ROI[:,:,anno_i]).keys()))
    NGTDMFeaturesNames_V = [s.replace('original_ngtdm', 'NGTDM') + "_Vessel"    for s in NGTDMFeaturesNames_V]

    GLDMFeaturesNames_O  = list(filter(lambda k: k.startswith("original_"), gldmFeaturesExtractor.execute( Vesselimage[:,:,anno_i], Outer_ROI[:,:,anno_i]).keys()))
    GLDMFeaturesNames_O  = [s.replace('original_gldm', 'GLDM') + "_OuterWall" for s in GLDMFeaturesNames_O]
    GLDMFeaturesNames_V  = list(filter(lambda k: k.startswith("original_"), gldmFeaturesExtractor.execute( Vesselimage[:,:,anno_i],Vessel_ROI[:,:,anno_i]).keys()))
    GLDMFeaturesNames_V  = [s.replace('original_gldm', 'GLDM') + "_Vessel"    for s in GLDMFeaturesNames_V]


    ShapeFeaturesNames = np.concatenate((ShapeFeaturesNames_O, ShapeFeaturesNames_V), axis=0)
    HistFeaturesNames  = np.concatenate(( HistFeaturesNames_O,  HistFeaturesNames_V), axis=0)
    GLCMFeaturesNames  = np.concatenate(( GLCMFeaturesNames_O,  GLCMFeaturesNames_V), axis=0)
    GLSZMFeaturesNames = np.concatenate((GLSZMFeaturesNames_O, GLSZMFeaturesNames_V), axis=0)
    GLRLMFeaturesNames = np.concatenate((GLRLMFeaturesNames_O, GLRLMFeaturesNames_V), axis=0)
    NGTDMFeaturesNames = np.concatenate((NGTDMFeaturesNames_O, NGTDMFeaturesNames_V), axis=0)
    GLDMFeaturesNames  = np.concatenate(( GLDMFeaturesNames_O,  GLDMFeaturesNames_V), axis=0)

    FeatureNames = np.concatenate((ShapeFeaturesNames, HistFeaturesNames, GLCMFeaturesNames, GLSZMFeaturesNames, GLRLMFeaturesNames, NGTDMFeaturesNames, GLDMFeaturesNames), axis=0)

    # Set Feature storage
    ShapeFeatureStorage = np.array(ShapeFeatureStorage).squeeze()
    HistFeaturesStorage = np.array(HistFeaturesStorage).squeeze()
    GLCMFeaturesStorage = np.array(GLCMFeaturesStorage).squeeze()
    GLSZMFeaturesStorage = np.array(GLSZMFeaturesStorage).squeeze()
    GLRLMFeaturesStorage = np.array(GLRLMFeaturesStorage).squeeze()
    NGTDMFeaturesStorage = np.array(NGTDMFeaturesStorage).squeeze()
    GLDMFeaturesStorage = np.array(GLDMFeaturesStorage).squeeze()

    TotalFeatures = np.concatenate((ShapeFeatureStorage, HistFeaturesStorage, GLCMFeaturesStorage, GLSZMFeaturesStorage, GLRLMFeaturesStorage, NGTDMFeaturesStorage, GLDMFeaturesStorage), axis=1)

    # Feature save
    np.savez(f"./feature/Radiomics.npz", TotalFeatures=TotalFeatures,FeatureNames=FeatureNames, label=label, p_info=p_info)

    data_df = pd.DataFrame(TotalFeatures, columns=FeatureNames) # converting to DataFrame
    data_df.to_csv(f"./feature/Radiomics.csv", index=False)