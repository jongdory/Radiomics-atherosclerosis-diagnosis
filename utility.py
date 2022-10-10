import radiomics
import numpy

def ShapeFeaturesExtractor(img, roi, roiNum):
    shapeFeature = radiomics.shape2D.RadiomicsShape2D(img, roi)
    shapeFeatureDict = {}
    
    shapeFeatureDict['Shape_MeshSurface_' + roiNum]            = shapeFeature.getMeshSurfaceFeatureValue()                # MeshSurface
    shapeFeatureDict['Shape_Perimeter_' + roiNum]              = shapeFeature.getPerimeterFeatureValue()                  # Perimeter
    shapeFeatureDict['Shape_PerimeterSurfaceRatio_' + roiNum]  = shapeFeature.getPerimeterSurfaceRatioFeatureValue()      # PerimeterSurfaceRatio
    shapeFeatureDict['Shape_Sphericity_' + roiNum]             = shapeFeature.getSphericityFeatureValue()                 # Sphericity
    shapeFeatureDict['Shape_SphericalDisproportion_' + roiNum] = shapeFeature.getSphericalDisproportionFeatureValue()     # SphericalDisproportion
    shapeFeatureDict['Shape_MaximumDiameter_' + roiNum]        = shapeFeature.getMaximumDiameterFeatureValue()            # MaximumDiameter
    shapeFeatureDict['Shape_MajorAxisLength_' + roiNum]        = shapeFeature.getMajorAxisLengthFeatureValue()            # MajorAxisLength
    shapeFeatureDict['Shape_MinorAxisLength_' + roiNum]        = shapeFeature.getMinorAxisLengthFeatureValue()            # MinorAxisLength
    shapeFeatureDict['Shape_Elongation_' + roiNum]             = shapeFeature.getElongationFeatureValue()                 # Elongation
    
    return shapeFeatureDict

def firstOrderFeaturesExtractor(img, roi, roiNum):
    firstOrderFeatures = radiomics.firstorder.RadiomicsFirstOrder(img, roi, binwidth=128, verbose=True, interpolator=None)
    firstOrderFeatures._initCalculation()
    firstOrderFeaturesDict = {}
    
    firstOrderFeaturesDict['FO_Energy_' + roiNum]                      = firstOrderFeatures.getEnergyFeatureValue()                               # Energy
    firstOrderFeaturesDict['FO_Entropy_' + roiNum]                     = firstOrderFeatures.getEntropyFeatureValue()                              # Entropy
    firstOrderFeaturesDict['FO_Minimum_' + roiNum]                     = firstOrderFeatures.getMinimumFeatureValue()                              # Minimum
    firstOrderFeaturesDict['FO_Maximum_' + roiNum]                     = firstOrderFeatures.getMaximumFeatureValue()                              # Maximum
    firstOrderFeaturesDict['FO_Mean_' + roiNum]                        = firstOrderFeatures.getMeanFeatureValue()                                 # Mean
    firstOrderFeaturesDict['FO_Variance_' + roiNum]                    = firstOrderFeatures.getVarianceFeatureValue()                             # Variance
    firstOrderFeaturesDict['FO_Standard_Deviation_' + roiNum]          = firstOrderFeatures.getStandardDeviationFeatureValue()                    # Standard deviation
    firstOrderFeaturesDict['FO_2.5th_Percentile_'  + roiNum]           = numpy.nanpercentile(firstOrderFeatures.targetVoxelArray, 2.5 , axis=1)   # 2.5th  Percentile
    firstOrderFeaturesDict['FO_25th_Percentile_'   + roiNum]           = numpy.nanpercentile(firstOrderFeatures.targetVoxelArray, 25  , axis=1)   # 25th   Percentile
    firstOrderFeaturesDict['FO_50th_Percentile_'   + roiNum]           = numpy.nanpercentile(firstOrderFeatures.targetVoxelArray, 50  , axis=1)   # 50th   Percentile
    firstOrderFeaturesDict['FO_75th_Percentile_'   + roiNum]           = numpy.nanpercentile(firstOrderFeatures.targetVoxelArray, 75  , axis=1)   # 75th   Percentile
    firstOrderFeaturesDict['FO_97.5th_Percentile_' + roiNum]           = numpy.nanpercentile(firstOrderFeatures.targetVoxelArray, 97.5, axis=1)   # 97.5th Percentile
    firstOrderFeaturesDict['FO_Inter_quartile_Range_' + roiNum]        = firstOrderFeatures.getInterquartileRangeFeatureValue()                   # Inter quartile range
    firstOrderFeaturesDict['FO_Range_' + roiNum]                       = firstOrderFeatures.getRangeFeatureValue()                                # Range
    firstOrderFeaturesDict['FO_MeanAbsoluteDeviation_' + roiNum]       = firstOrderFeatures.getMeanAbsoluteDeviationFeatureValue()                # MeanAbsoluteDeviation
    firstOrderFeaturesDict['FO_RobustMeanAbsoluteDeviation_' + roiNum] = firstOrderFeatures.getRobustMeanAbsoluteDeviationFeatureValue()          # RobustMeanAbsoluteDeviation
    firstOrderFeaturesDict['FO_RootMeanSquaredDeviation_' + roiNum]    = firstOrderFeatures.getRootMeanSquaredFeatureValue()                      # RootMeanSquared
    firstOrderFeaturesDict['FO_Skewness_' + roiNum]                    = firstOrderFeatures.getSkewnessFeatureValue()                             # Skewness
    firstOrderFeaturesDict['FO_Kurtosis_' + roiNum]                    = firstOrderFeatures.getKurtosisFeatureValue()                             # Kurtosis
    firstOrderFeaturesDict['FO_Uniformity_' + roiNum]                  = firstOrderFeatures.getUniformityFeatureValue()                           # Uniformity

    
    return firstOrderFeaturesDict
