distanceTransform() [1/2]
void cv::distanceTransform	(	InputArray 	src,
OutputArray 	dst,
OutputArray 	labels,
int 	distanceType,
int 	maskSize,
int 	labelType = DIST_LABEL_CCOMP 
)		
Python:
cv.distanceTransform(	src, distanceType, maskSize[, dst[, dstType]]	) ->	dst
cv.distanceTransformWithLabels(	src, distanceType, maskSize[, dst[, labels[, labelType]]]	) ->	dst, labels
#include <opencv2/imgproc.hpp>

Calculates the distance to the closest zero pixel for each pixel of the source image.

The function cv::distanceTransform calculates the approximate or precise distance from every binary image pixel to the nearest zero pixel. For zero image pixels, the distance will obviously be zero.

When maskSize == DIST_MASK_PRECISE and distanceType == DIST_L2 , the function runs the algorithm described in [69] . This algorithm is parallelized with the TBB library.

In other cases, the algorithm [28] is used. This means that for a pixel the function finds the shortest path to the nearest zero pixel consisting of basic shifts: horizontal, vertical, diagonal, or knight's move (the latest is available for a 5×5 mask). The overall distance is calculated as a sum of these basic distances. Since the distance function should be symmetric, all of the horizontal and vertical shifts must have the same cost (denoted as a ), all the diagonal shifts must have the same cost (denoted as b), and all knight's moves must have the same cost (denoted as c). For the DIST_C and DIST_L1 types, the distance is calculated precisely, whereas for DIST_L2 (Euclidean distance) the distance can be calculated only with a relative error (a 5×5 mask gives more accurate results). For a,b, and c, OpenCV uses the values suggested in the original paper:

DIST_L1: a = 1, b = 2
DIST_L2:
3 x 3: a=0.955, b=1.3693
5 x 5: a=1, b=1.4, c=2.1969
DIST_C: a = 1, b = 1
Typically, for a fast, coarse distance estimation DIST_L2, a 3×3 mask is used. For a more accurate distance estimation DIST_L2, a 5×5 mask or the precise algorithm is used. Note that both the precise and the approximate algorithms are linear on the number of pixels.

This variant of the function does not only compute the minimum distance for each pixel (x,y) but also identifies the nearest connected component consisting of zero pixels (labelType==DIST_LABEL_CCOMP) or the nearest zero pixel (labelType==DIST_LABEL_PIXEL). Index of the component/pixel is stored in labels(x, y). When labelType==DIST_LABEL_CCOMP, the function automatically finds connected components of zero pixels in the input image and marks them with distinct labels. When labelType==DIST_LABEL_PIXEL, the function scans through the input image and marks all the zero pixels with distinct labels.

In this mode, the complexity is still linear. That is, the function provides a very fast way to compute the Voronoi diagram for a binary image. Currently, the second variant can use only the approximate distance transform algorithm, i.e. maskSize=DIST_MASK_PRECISE is not supported yet.

Parameters
src	8-bit, single-channel (binary) source image.
dst	Output image with calculated distances. It is a 8-bit or 32-bit floating-point, single-channel image of the same size as src.
labels	Output 2D array of labels (the discrete Voronoi diagram). It has the type CV_32SC1 and the same size as src.
distanceType	Type of distance, see DistanceTypes
maskSize	Size of the distance transform mask, see DistanceTransformMasks. DIST_MASK_PRECISE is not supported by this variant. In case of the DIST_L1 or DIST_C distance type, the parameter is forced to 3 because a 3×3 mask gives the same result as 5×5 or any larger aperture.
labelType	Type of the label array to build, see DistanceTransformLabelTypes.
Examples:
samples/cpp/distrans.cpp.