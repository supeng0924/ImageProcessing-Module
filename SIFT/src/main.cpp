#include <opencv2\opencv.hpp>
#include <opencv2\xfeatures2d.hpp>//note:if do not include "opencv/include" cmake send error.
//must include "opencv/include" "opencv/include/opencv" "opencv/include/opencv2" the three all of directory
//#include <opencv2\highgui.hpp>
#include <iostream>
#include <chrono>
#include "app.h"
#include "image.h"
#include "counttime.h"
#include <opencv2\features2d.hpp>
using namespace std::chrono;
using namespace cv::xfeatures2d;
using namespace cv;

typedef steady_clock Clock;
//#define USE_OPENCV_SIFT    //use opencv library function sift
//#define USE_SELF_W_SIFT    //use the sift programmed by myself 
#define USE_SIFT_MATCH     //use the sift programmed detect and use match

#undef main  //#undef cancel the preprocessor definition from the previous program
int main(int, char**)
{
	//SIFT algorithm implement by using opencv library function
	cv::Mat Lena_total = cv::imread("../Lena_total.jpg", 0);
	cv::Mat origin_im = cv::imread("../Lena.jpg", 0);
	if (origin_im.empty()|| Lena_total.empty())
	{
		std::cout << "read failed" << std::endl;
		return -1;
	}
	cv::imshow("lena_origin",origin_im);
	cv::imshow("lena_total", Lena_total);
	//Mat dst;
	//cv::resize(Lena_total, dst, cv::Size(Lena_total.size() /2), 0, 0, cv::INTER_NEAREST);



#ifdef USE_OPENCV_SIFT
	//SIFT detector
	int numFeatures = 100;
	Ptr<SIFT> detector = SIFT::create(numFeatures);
	std::vector<KeyPoint> keypoints;

	detector->detect(origin_im, keypoints, Mat());
	std::cout << keypoints.size() << std::endl;

	//draw sift keypoints
	cv::Mat keypoint_img;
	drawKeypoints(origin_im, keypoints, keypoint_img, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("keypoint_img", keypoint_img);
	cv::Size pp = origin_im.size() / 2;
	std::cout << origin_im.size() << "  " << pp << std::endl;
	//Mat descriptors1;
	//detector->detectAndCompute(origin_im, Mat(), keypoints, descriptors1);
	//std::cout << descriptors1.ptr<float>(0)[0] <<" "<<descriptors1.ptr<float>(0)[1] << std::endl;
	//std::vector<double> sig()
#endif // USE_OPENCV_SIFT

#ifdef USE_SELF_W_SIFT
	cv::Mat result_img = Lena_total.clone();
	SiftAL sift_own;
	cv::Mat base;

	base =sift_own.createInitalImage(Lena_total, true,1.6);
	//imshow("resutl", base);

	std::vector<cv::Mat> gpyr;
	int nOctaves = log2(min(base.size().width, base.size().height)) - 2 + 1;
	sift_own.buildGaussPyramid(base, gpyr, nOctaves);
	std::vector<cv::Mat> dogpyr;
	sift_own.buildDoGPyramid(gpyr, dogpyr);

	std::vector<cv::KeyPoint> keypoints;
	sift_own.findScaleExtrema(gpyr, dogpyr, keypoints);
	//sift size
	int dsize = 4 * 4 * 8;
	cv::Mat descriptors;
	descriptors.create(keypoints.size(), dsize, CV_32F);
	int noctavelay = sift_own.nOctaveLayers;
	int firstoctave = -1;
	sift_own.calcDescriptors(gpyr, keypoints, descriptors, noctavelay, firstoctave);

	for (int k = 0; k < keypoints.size(); k++)
	{
		cv::Point pt;
		pt.x= (int)(keypoints[k].pt.x/2);
		pt.y = (int)(keypoints[k].pt.y/2);
		if (pt.x<0 || pt.x>result_img.cols || pt.y < 0 || pt.y>result_img.rows)
		{
			std::cout << "out of range" << std::endl;
			return -1;
		}
		circle(result_img, pt, 1, Scalar(255, 0, 0), 3);
	}
	imshow("result_img", result_img);
	std::cout << "ok" << std::endl;

	Mat img_matches;

#endif // USE_SELF_W_SIFT
#ifdef USE_SIFT_MATCH
	cv::Mat result_img = Lena_total.clone();
	
	SiftAL sift_ori;

	cv::Mat base_ori;
	base_ori = sift_ori.createInitalImage(Lena_total, true, 1.6);

	std::vector<cv::Mat> gpyr_ori;
	std::vector<cv::Mat> dogpyr_ori;
	std::vector<cv::KeyPoint> keypoints_ori;
	cv::Mat descriptors_ori;
	int dsize_ori = 4 * 4 * 8;
	int layers_to_ori = sift_ori.nOctaveLayers;
	int firstoctave_ori = -1;
	int nOctaves_ori = log2(min(base_ori.size().width, base_ori.size().height)) - 2 + 1;

	sift_ori.buildGaussPyramid(base_ori, gpyr_ori, nOctaves_ori);
	sift_ori.buildDoGPyramid(gpyr_ori, dogpyr_ori);
	sift_ori.findScaleExtrema(gpyr_ori, dogpyr_ori, keypoints_ori);
	descriptors_ori.create(keypoints_ori.size(), dsize_ori, CV_32F);
	sift_ori.calcDescriptors(gpyr_ori, keypoints_ori, descriptors_ori, layers_to_ori, firstoctave_ori);


	Mat descri_ori_filter;
	std::vector<cv::KeyPoint> keypoints_ori_filter;
	sift_ori.DescriptorFilter(keypoints_ori, keypoints_ori_filter, descriptors_ori, descri_ori_filter);





	SiftAL sift_part;
	cv::Mat base_part;
	base_part = sift_part.createInitalImage(origin_im, true, 1.6);

	std::vector<cv::Mat> gpyr_part;
	std::vector<cv::Mat> dogpyr_part;
	std::vector<cv::KeyPoint> keypoints_part;
	cv::Mat descriptors_part;
	int dsize_part = 4 * 4 * 8;
	int layers_to_part = sift_part.nOctaveLayers;
	int firstoctave_part = -1;
	int nOctaves_part = log2(min(base_part.size().width, base_part.size().height)) - 2 + 1;

	sift_part.buildGaussPyramid(base_part, gpyr_part, nOctaves_part);
	sift_part.buildDoGPyramid(gpyr_part, dogpyr_part);
	sift_part.findScaleExtrema(gpyr_part, dogpyr_part, keypoints_part);
	descriptors_part.create(keypoints_part.size(), dsize_part, CV_32F);
	sift_part.calcDescriptors(gpyr_part, keypoints_part, descriptors_part, layers_to_part, firstoctave_part);



	Mat descri_part_filter;
	std::vector<cv::KeyPoint> keypoints_part_filter;
	sift_ori.DescriptorFilter(keypoints_part, keypoints_part_filter, descriptors_part, descri_part_filter);






	Mat img_matches;
	Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create("BruteForce");
	std::vector<DMatch> matches;
	descriptor_matcher->match(descri_part_filter, descri_ori_filter, matches);

	drawMatches(origin_im, keypoints_part_filter, Lena_total, keypoints_ori_filter, matches, img_matches, Scalar::all(-1), CV_RGB(255, 255, 255), Mat(), 4);

	imshow("Mathc", img_matches);

	//std::cout << keypoints_ori.size() << "  " << keypoints_part.size() << std::endl;

/*
	for (int k = 0; k < keypoints.size(); k++)
	{
		cv::Point pt;
		pt.x = (int)(keypoints[k].pt.x / 2);
		pt.y = (int)(keypoints[k].pt.y / 2);
		if (pt.x<0 || pt.x>result_img.cols || pt.y < 0 || pt.y>result_img.rows)
		{
			std::cout << "out of range" << std::endl;
			return -1;
		}
		circle(result_img, pt, 1, Scalar(255, 0, 0), 3);
	}
	imshow("result_img", result_img);
	std::cout << "ok" << std::endl;

	Mat img_matches;*/

#endif // USE_SIFT_MATCH

	std::cout << "ok" << std::endl;
	cv::waitKey(0);
	



	return 0;
}



