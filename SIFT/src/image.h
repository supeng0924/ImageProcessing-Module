#pragma once
#include <opencv2\opencv.hpp>
//#define DEBUG_CALC_DESCRIPTORS
//#define DEBUG_FIND_EXTREMA_MODE
typedef cv::Point3_<uchar>  Pixel;
class image {
public:
	void image_init(cv::Mat&);
	void ImageResize(void);
	void SetScale(void);
	void ForeachAl(void);
	void PointAl(void);
private:
	cv::Mat image_res;
};
void complicatedThreshold(Pixel &pixel);
struct Operator
{
	void operator ()(Pixel &pixel, const int * position) const
	{
		//perform a simple threshold operation
		complicatedThreshold(pixel);
	}
};
/*
** 
** sift algorithm implement
** include four step
** 1: scale-space extrema detection
** 2: keypoint localization
** 3: orientaton assignment
** 4: keypoint descriptor
**
*/
class SiftAL {
public:
	cv::Mat image;
	SiftAL(int _layers = 3, double _sigma = 1.6);
	static cv::Mat createInitalImage(const cv::Mat& img, bool doubleImageSize, float sigma);
	void buildGaussPyramid(const cv::Mat& base, std::vector<cv::Mat>& pyr, int nOctaves) const;
	void buildDoGPyramid(const std::vector<cv::Mat>& gpyr, std::vector<cv::Mat>& dogpyr) const;
	void findScaleExtrema(const std::vector < cv::Mat>& gauss_pyr, const std::vector<cv::Mat>& dog_pyr, std::vector<cv::KeyPoint>& keypoints);
	bool adjustLocalExtrema(const std::vector<cv::Mat>& dog_pyr, cv::KeyPoint& kpt, int octv, int& layer, int& r, int& c, float contrastThreshold, float edgeThreshold);
	static float calcOrientationHist(const cv::Mat& img, cv::Point pt, int radius, float sigma, float *hist, int n);
	static void calcDescriptors(const std::vector<cv::Mat>& gpyr, const std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int nOctaveLayers,int firstOctave);
	static void calcSIFTDescriptor(const cv::Mat& img, cv::Point2f ptf, float ori, float scl, int d, int n, float *dst);
	void DescriptorFilter(std::vector<cv::KeyPoint>& keypoints_ori, std::vector<cv::KeyPoint>& keypoints_filter, cv::Mat& descriptors_ori,cv::Mat& descriptors_filter);

	int nOctaveLayers;   //how many pictures in a octave 
	double sigma;      //variance of gaussian distribution 
	//double contrastThreshold  0.04
	//int edgeThreshold  //the ratio of two eigenvalues 

};
#define SIFT_ORI_HIST_BINS 36
#define SIFT_IMG_BORDER 5        //image border
#define SIFT_FIXPT_SCALE 1
#define SIFT_MAX_INTERP_STEPS 5  //the number of iterations of the loop
#define SIFT_INIT_SIGMA 0.5      //init scale 
#define sift_wt short
#define SIFT_ORI_SIG_FCTR 1.5f  //oritention sigma
#define SIFT_ORI_RADIUS (3*SIFT_ORI_SIG_FCTR) //local structure radius
#define SIFT_ORI_PEAK_RATIO 0.8 //histogram auxiliary direction = main orientation peak ration
#define SIFT_DESCR_WIDTH      4 //discriptors width
#define SIFT_DESCR_HIST_BINS  8 //BINS
#define SIFT_DESCR_SCL_FCTR   3.0 //3sigma
#define SIFT_DESCR_MAG_THR  0.2 
//#define SIFT_INT_DESCR_FCTR 512.0
#define SIFT_INT_DESCR_FCTR 256.0

