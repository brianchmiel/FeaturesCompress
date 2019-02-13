

import matplotlib.pyplot as plt


if __name__ == '__main__':
    Bitwidth = [2,3,4,5,6,8]
    eyeAll = [10.08, 10.51, 37.94, 89.04,91.89, 92.29]
    pcaAllOneDynamicRange = [9.88, 10.205, 23.865, 75.842, 89.635, 92.214]
    pcaAllDynamicRangePerLayer = [11.784,58.216, 87.467,91.523, 91.953,92.318]
    pcaAllDynamicRangePerLayerTruncated0_95 = [11.61,55.378, 85.234, 90.143, 91.055,92.09]
    pcaAllDynamicRangePerLayerTruncated0_99 = [12.33, 57.552, 86.823, 91.055, 92.083,92.57]
    pcaAllDynamicRangePerLayerTruncated0_8 = [XX, 22.73, 23.6, 23.08, 22.9, XX ]
    pcaAllDynamicRangePerLayerTruncated0_9 = [XX, 49.03, 78.56, 83.97, 84.99, XX]

    pcaAllDynamicRangePerLayerBlockSize1 = [17.72, 77.43, 90.46, 91.85, 92.16, 92.21 ]
    pcaAllDynamicRangePerLayerBlockSize2 = [13.04, 65.85, 88.52, 92.09, 91.95, 92.25]




    plt.plot(Bitwidth,eyeAll,color='blue')
    # plt.plot(Bitwidth,pcaAllOneDynamicRange,color='red' )
    plt.plot(Bitwidth,pcaAllDynamicRangePerLayer, color='green')
    plt.plot(Bitwidth, pcaAllDynamicRangePerLayerTruncated0_95, color='black')
 #   plt.plot(Bitwidth, pcaAllDynamicRangePerLayerTruncated0_99, color='yellow')
    plt.plot(Bitwidth, pcaAllDynamicRangePerLayerBlockSize1, color='cyan')
    plt.xlabel('Bitwidth')
    plt.ylabel('Top 1 Acc')
    plt.gca().legend(('Without projection', 'PCA Projection - block size = 4','PCA Projection - block size = 4. Prunning 95%','PCA Projection - block size = 1'))
    plt.show()


