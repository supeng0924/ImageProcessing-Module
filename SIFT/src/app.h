#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <Kinect.h>
#include <opencv2\opencv.hpp>
//some often used stl header files
#include <iostream>
#include <memory>
#include <vector>

//use data type macro definition
#define COLOR_TYPE  //get color data 
//#define DEPTH_TYPE  //get depth data
//#define MAP_TYPE    //get map   data(simply implement)


//some useful typedefs for explicit type sizes
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;
typedef char int8;
typedef short int16;
typedef int int32;
typedef long long int64;

//safe way of deleting a COM object
template<typename T>
void SafeRelease(T& ptr) { if (ptr) { ptr->Release(); ptr = nullptr; } }


/*
**refer url:https://www.youtube.com/watch?v=L1Kgm4S8c90
**kinect sensor init
**kinect get color + depth data
*/
class App
{
public:
	void Init();//init sensor
	bool Update();//update color and depth data
	void OpencvShow();//convert to mat type and show
	void Shutdown();//close sensor and release memory
	void PrintTest(void);//no use, just output statement
	cv::Mat GetColorImage(void);//get color bgra 1080p image

private:

	//kinect sensor
	IKinectSensor* m_sensor = nullptr;
	//depth data
	IDepthFrameReader* m_depthFrameReader= nullptr;
	uint16 *m_depthBuffer;
	int m_depthWidth = 0, m_depthHeight = 0;
	//color data
	IColorFrameReader* m_colorFrameReader = nullptr;
	uint8 *m_colorBuffer;
	int m_colorWidth = 0, m_colorHeight = 0;
	//map
	ICoordinateMapper* m_coordinateMapper = nullptr;
	ColorSpacePoint* m_colorSpacePoints = nullptr;

	//color image
	cv::Mat color_image;
};





