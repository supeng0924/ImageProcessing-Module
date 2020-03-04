#include "app.h"
#include <opencv2\opencv.hpp>
void App::Init()
{
	//put initialization stuff here

	//get the kinect sensor
	HRESULT hr;
	hr = GetDefaultKinectSensor(&m_sensor);
	if (FAILED(hr))
	{
		std::cout << "failed to find the kinect sensor!" << std::endl;
		exit(10);
	}
	m_sensor->Open();


	//***************************depth***********************************
#ifdef DEPTH_TYPE
	//get the depth frame source
	IDepthFrameSource* depthFrameSource;
	hr = m_sensor->get_DepthFrameSource(&depthFrameSource);
	if (FAILED(hr))
	{
		std::cout << "failed to get the depth frame source" << std::endl;
		exit(10);
	}
	//depth description
	IFrameDescription *frameDesc;
	depthFrameSource->get_FrameDescription(&frameDesc);
	frameDesc->get_Width(&m_depthWidth);
	frameDesc->get_Height(&m_depthHeight);



	//get the depth frame reader
	hr = depthFrameSource->OpenReader(&m_depthFrameReader);
	if (FAILED(hr))
	{
		std::cout << "failed to open the depth frame reader" << std::endl;
		exit(10);
	}
	//release depth frame source
	SafeRelease(depthFrameSource);

	//allocate depth buffer
	m_depthBuffer = new uint16[m_depthWidth * m_depthHeight];
#endif // DEPTH_TYPE
	//*****************************************************************



	//***************************color***********************************
#ifdef COLOR_TYPE
	//get the color frame source
	IColorFrameSource* colorFrameSource;
	hr = m_sensor->get_ColorFrameSource(&colorFrameSource);
	if (FAILED(hr))
	{
		std::cout << "failed to get the color frame source" << std::endl;
		exit(10);
	}
	//color descriptor
	IFrameDescription *frameDescColor;
	colorFrameSource->get_FrameDescription(&frameDescColor);
	frameDescColor->get_Width(&m_colorWidth);
	frameDescColor->get_Height(&m_colorHeight);

	//get the color frame reader
	hr = colorFrameSource->OpenReader(&m_colorFrameReader);
	if (FAILED(hr))
	{
		std::cout << "failed to open the color frame reader" << std::endl;
		exit(10);
	}

	//release color frame source
	SafeRelease(colorFrameSource);
	//allocate color buffer
	m_colorBuffer = new uint8[m_colorWidth * m_colorHeight * 4];
#endif // COLOR_TYPE
	//***********************************************************************

	//*****************************map***********************************
#ifdef MAP_TYPE
	//get the coordinate mapper
	hr = m_sensor->get_CoordinateMapper(&m_coordinateMapper);
	if (FAILED(hr))
	{
		std::cout << "failed to get the color coordinate mapper" << std::endl;
		exit(10);
	}
	//allocate a buffer of color space points
	m_colorSpacePoints = new ColorSpacePoint[m_depthWidth * m_depthHeight];
#endif // MAP_TYPE
	//**********************************************************************


}
bool App::Update()
{
	HRESULT hr;//COM error check

#ifdef DEPTH_TYPE
	//depth data stuff
	IDepthFrame* depthFrame;
	hr = m_depthFrameReader->AcquireLatestFrame(&depthFrame);
	hr = depthFrame->CopyFrameDataToArray(m_depthWidth * m_depthHeight, m_depthBuffer);
	if (FAILED(hr))
	{
		SafeRelease(depthFrame);
		std::cout << "something wrong while copying depth !" << std::endl;
		return false;
	}
	SafeRelease(depthFrame);
#endif // DEPTH_TYPE

#ifdef COLOR_TYPE
	//color data stuff
	IColorFrame* colorFrame;
	hr = m_colorFrameReader->AcquireLatestFrame(&colorFrame);
	if (FAILED(hr))
		return false;
	hr = colorFrame->CopyConvertedFrameDataToArray(m_colorWidth * m_colorHeight * 4, m_colorBuffer, ColorImageFormat_Bgra);
	if (FAILED(hr))
	{
		SafeRelease(colorFrame);
		std::cout << "something wrong while copying color!" << std::endl;
		return false;
	}
	std::cout << "Coping  mapping!" << std::endl;
	SafeRelease(colorFrame);
#endif // COLOR_TYPE
	OpencvShow();
	return true;
}
void App::OpencvShow()
{
	
#ifdef DEPTH_TYPE
	//copy depth data to the screen
	int po_dep_i = 0;
	cv::Mat dep_image = cv::Mat(m_depthHeight, m_depthWidth, CV_16UC1);
	po_dep_i = 0;
	for (int i = 0; i < dep_image.rows; i++)
	{
		uint16 *data = dep_image.ptr<uint16>(i);
		for (int j = 0; j < dep_image.cols; j++)
		{
			data[j] = *(m_depthBuffer + po_dep_i);
			po_dep_i++;
		}
	}
	cv::imshow("dep_image", dep_image);
#endif // DEPTH_TYPE

#ifdef COLOR_TYPE
	//copy color data to the screen
	color_image = cv::Mat(m_colorHeight, m_colorWidth, CV_8UC3);
	int po_col_i = 0;
	for (int i = 0; i < color_image.rows; i++)
	{
		uint8 *data = color_image.ptr<uint8>(i);
		for (int j = 0; j < color_image.cols; j++)
		{
			int j_3j = j * 3;
			data[j_3j] = *(m_colorBuffer + po_col_i); po_col_i++;
			data[j_3j+1] = *(m_colorBuffer + po_col_i); po_col_i++;
			data[j_3j+2] = *(m_colorBuffer + po_col_i); po_col_i++; po_col_i++;
		}
	}
	//cv::imshow("color_image", color_image);
#endif // COLOR_TYPE

#ifdef MAP_TYPE
	HRESULT hr;//COM error check
	hr = m_coordinateMapper->MapDepthFrameToColorSpace(m_depthWidth * m_depthHeight,
		m_depthBuffer, m_depthWidth * m_depthHeight,
		m_colorSpacePoints);
	if (FAILED(hr))
	{
		std::cout << "failed to map the depth frame to color space!" << std::endl;
		return;
	}
	cv::Mat map_image = cv::Mat(m_depthHeight, m_depthWidth, CV_8UC3);
	int po_map_i = 0;
	for (int i = 0; i < map_image.rows; i++)
	{
		uint8 *data = map_image.ptr<uint8>(i);
		for (int j = 0; j < map_image.cols; j++)
		{
			ColorSpacePoint csp = m_colorSpacePoints[po_map_i];
			po_map_i++;
			int ix = (int)csp.X;
			int iy = (int)csp.Y;

			if (ix > m_colorWidth - 2) { ix = m_colorWidth - 1; }
			if (iy > m_colorHeight - 2) { iy = m_colorHeight - 1; }
			if (ix < 0) { ix = 0; }
			if (iy < 0) { iy = 0; }
			int po_pos = (iy*m_colorWidth + ix) * 4;

			int j_3j = j * 3;
			data[j_3j] = *(m_colorBuffer + po_pos);
			data[j_3j + 1] = *(m_colorBuffer + po_pos + 1);
			data[j_3j + 2] = *(m_colorBuffer + po_pos + 2);

		}
	}
	cv::imshow("map_image", map_image);
#endif // MAP_TYPE
	
}
void App::Shutdown()
{
	//put cleaning up stuff here

	//release the depth stuff
#ifdef DEPTH_TYPE
	delete[] m_depthBuffer;
	SafeRelease(m_depthFrameReader);
#endif // DEPTH_TYPE

	//release the color stuff
#ifdef COLOR_TYPE
	delete[] m_colorBuffer;
	SafeRelease(m_colorFrameReader);
#endif // COLOR_TYPE

	//close kinect sensor
	m_sensor->Close();
	SafeRelease(m_sensor);	
	
}
void App::PrintTest(void) 
{
	std::cout << "wait get data" << std::endl;
}
cv::Mat App::GetColorImage(void)
{
	return color_image;
}