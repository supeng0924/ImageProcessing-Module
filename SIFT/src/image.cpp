#include "image.h"
#include <opencv2\opencv.hpp>
#include <opencv2\core\hal\hal.hpp>
#include <iostream>
void image::image_init(cv::Mat& input_im)
{
	cv::resize(input_im, image_res, cv::Size(), 0.5, 0.5);
	cv::imshow("image_res", image_res);
}
void image::ForeachAl(void)
{
	image_res.forEach<Pixel>(Operator());	
}
void image::PointAl(void)
{
	for (int i = 0; i < image_res.rows; i++)
	{
		uchar *data = image_res.ptr<uchar>(i);
		for (int j = 0; j < image_res.cols; j++)
		{
			int jj = j * 3;
			if (pow(double(data[jj]) / 10, 2.5) > 100)
			{
				data[jj] = 255;
				data[jj + 1] = 255;
				data[jj + 2] = 255;
			}
			else
			{
				data[jj] = 0;
				data[jj + 1] = 0;
				data[jj + 2] = 0;
			}
		}
	}
}
void complicatedThreshold(Pixel &pixel)
{
	if (pow(double(pixel.x) / 10, 2.5) > 100)
	{
		pixel.x = 255;
		pixel.y = 255;
		pixel.z = 255;
	}
	else
	{
		pixel.x = 0;
		pixel.y = 0;
		pixel.z = 0;
	}
}

//class SiftAL member functions
SiftAL::SiftAL(int _layers, double _sigma)
{
	nOctaveLayers = _layers;
	sigma = _sigma;
}
cv::Mat SiftAL::createInitalImage(const cv::Mat& img, bool doubleImageSize, float sigma)
{
	cv::Mat gray, gray_fpt;
	if (img.channels() == 3 || img.channels() == 4)
	{
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	}
	else
		img.copyTo(gray);
	gray.convertTo(gray_fpt, cv::DataType<short>::type, SIFT_FIXPT_SCALE, 0);
	float sig_diff;
	if (doubleImageSize)//need to expend the image size of width and height
	{
		//SIFT_INIT_SIGMA is 0.5.that is the input image sigma scale
		//if double image size the scale is SIFT_INIT_SIGMA*2
		//from the σ1->σ2 : σ_dif(sig_diff)=sqrt(σ2-σ1)
		sig_diff = sqrtf(std::max((sigma*sigma - SIFT_INIT_SIGMA*SIFT_INIT_SIGMA * 4), 0.01));
		//dbl:double image 
		cv::Mat dbl;
		cv::resize(gray_fpt, dbl, gray.size()*2, 0, 0, cv::INTER_LINEAR);
		cv::GaussianBlur(dbl, dbl, cv::Size(), sig_diff, sig_diff);
		return dbl;
	}
	else
	{
		sig_diff = sqrtf(std::max((sigma*sigma - SIFT_INIT_SIGMA*SIFT_INIT_SIGMA), 0.01));
		cv::GaussianBlur(gray_fpt, gray_fpt, cv::Size(), sig_diff, sig_diff);
		return gray_fpt;
	}
}

void SiftAL::buildGaussPyramid(const cv::Mat& base, std::vector<cv::Mat>& pyr, int nOctaves) const
{
	std::vector<double> sig(nOctaveLayers + 3);  //variance vector
	pyr.resize(nOctaves*(nOctaveLayers + 3));    //the difference of gaussian pyramid size

	sig[0] = sigma;
	double k = pow(2, 1. / nOctaveLayers);
	for (int i = 1; i < nOctaveLayers + 3; i++)
	{
		double sig_prev = pow(k, i - 1)*sigma;
		double sig_total = sig_prev*k;

		//scale coordinates
		//sig=sqrt( ((k^s)*sigma)^2-((k^(s-1))*sigma)^2 )
		sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
	}

	for (int o = 0; o < nOctaves; o++)
	{
		//total  nOctaveLayers+3 images
		for (int i = 0; i < nOctaveLayers + 3; i++)
		{
			//Octave o layer i image
			cv::Mat& dst = pyr[o*(nOctaveLayers + 3) + i];

			//the first image use the origin image
			if (o == 0 && i == 0)
				dst = base;
			//Octave o the first image, countdown last octave the third image from the bottom 
			else if (i == 0)
			{
				const cv::Mat& src = pyr[(o - 1)*(nOctaveLayers + 3) + nOctaveLayers];
				cv::resize(src, dst, src.size() / 2,0,0,cv::INTER_NEAREST);
			}
			//octave o image i use the i-1 image guassian
			else
			{
				const cv::Mat& src = pyr[o*(nOctaveLayers + 3) + i - 1];
				cv::GaussianBlur(src, dst, cv::Size(), sig[i], sig[i]);
			}
		}
	}

}
void SiftAL::buildDoGPyramid(const std::vector<cv::Mat>& gpyr, std::vector<cv::Mat>& dogpyr) const
{
	//compute the octaves in the pyramid
	int nOctaves = (int)gpyr.size() / (nOctaveLayers + 3);
	//DoG pyramid size  because of the image of DoG use two images of Gauss pyramid to subtract
	dogpyr.resize(nOctaves*(nOctaveLayers + 2));

	for (int o = 0; o < nOctaves; o++)
	{
		for (int i = 0; i < nOctaveLayers + 2; i++)
		{
			//the ith image of DoG = ith image of Gauss pyramid - i+1 image of Gauss pyramid
			const cv::Mat& src1 = gpyr[o*(nOctaveLayers + 3) + i];
			const cv::Mat& src2 = gpyr[o*(nOctaveLayers + 3) + i+1];
			cv::Mat& dst = dogpyr[o*(nOctaveLayers + 2) + i];
			cv::subtract(src1, src2, dst, cv::noArray(), CV_16S);
		}
	}
}

// detects features at extrema in DoG scale space. bad features are discarded
// based on contrast and ratio of principal curvatures
void SiftAL::findScaleExtrema(const std::vector < cv::Mat>& gauss_pyr, const std::vector<cv::Mat>& dog_pyr, std::vector<cv::KeyPoint>& keypoints)
{
	int nOctaves = (int)gauss_pyr.size() / (nOctaveLayers + 3);
	const int n = SIFT_ORI_HIST_BINS;

	cv::KeyPoint kpt;
	keypoints.clear();

	//set a threshold, be used to judge pixel value in the DoG scale image
	double contrastThreshold = 0.04;
	int threshold = cvFloor(0.5*contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);
	float hist[n];
	
	for (int o = 0; o < nOctaves; o++)
	{
		//total nOctaveLayers+2 layers.because compare last and next,so i total
		//is nOctaveLayers layers. start from 1 to  nOctaveLayers
		for (int i = 1; i < nOctaveLayers + 1; i++)
		{
			int idx = o * (nOctaveLayers + 2) + i;
			const cv::Mat& img = dog_pyr[idx];
			const cv::Mat& prev = dog_pyr[idx - 1];
			const cv::Mat& next = dog_pyr[idx + 1];
			int step = (int)img.step1();
			int rows = img.rows;
			int cols = img.cols;
			

			for (int r = SIFT_IMG_BORDER; r < rows - SIFT_IMG_BORDER; r++)
			{
				const short* currptr = img.ptr<short>(r);
				const short* prevptr = prev.ptr<short>(r);
				const short* nextptr = next.ptr<short>(r);
				
				for (int c = SIFT_IMG_BORDER; c < cols - SIFT_IMG_BORDER; c++)
				{
					int val = currptr[c];
					if (std::abs(val) > threshold &&
						(
						(
							val > 0 && val >= currptr[c - 1] && val >= currptr[c + 1] &&
							val >= currptr[c - step - 1] && val >= currptr[c - step] &&
							val >= currptr[c - step + 1] && val >= currptr[c + step - 1] &&
							val >= currptr[c + step] && val >= currptr[c + step + 1] &&
							val >= nextptr[c] && val >= nextptr[c - 1] &&
							val >= nextptr[c + 1] && val >= nextptr[c - step - 1] &&
							val >= nextptr[c - step] && val >= nextptr[c - step + 1] &&
							val >= nextptr[c + step - 1] && val >= nextptr[c + step] &&
							val >= nextptr[c + step + 1] && val >= prevptr[c] &&
							val >= prevptr[c - 1] && val >= prevptr[c + 1] &&
							val >= prevptr[c - step - 1] && val >= prevptr[c - step] &&
							val >= prevptr[c - step + 1] && val >= prevptr[c + step - 1] &&
							val >= prevptr[c + step] && val >= prevptr[c + step + 1]
							) ||
							(
								val < 0 && val <= currptr[c - 1] && val <= currptr[c + 1] &&
								val <= currptr[c - step - 1] && val <= currptr[c - step] &&
								val <= currptr[c - step + 1] && val <= currptr[c + step - 1] &&
								val <= currptr[c + step] && val <= currptr[c + step + 1] &&
								val <= nextptr[c] && val <= nextptr[c - 1] &&
								val <= nextptr[c + 1] && val <= nextptr[c - step - 1] &&
								val <= nextptr[c - step] && val <= nextptr[c - step + 1] &&
								val <= nextptr[c + step - 1] && val <= nextptr[c + step] &&
								val <= nextptr[c + step + 1] && val <= prevptr[c] &&
								val <= prevptr[c - 1] && val <= prevptr[c + 1] &&
								val <= prevptr[c - step - 1] && val <= prevptr[c - step] &&
								val <= prevptr[c - step + 1] && val <= prevptr[c + step - 1] &&
								val <= prevptr[c + step] && val <= prevptr[c + step + 1]
								)
							)
						)
					{
						int r1 = r, c1 = c, layer = i;
						float edgeThreshold = 10;
						
#ifdef DEBUG_FIND_EXTREMA_MODE
						std::cout << "find extrem point" << std::endl;
#endif // DEBUG_FIND_EXTREMA_MODE

						//if adjustLocalExtrema return false
						//it shows it is not the feature point
						//then it continue the circle
						if (!adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1, (float)contrastThreshold, (float)edgeThreshold))
						{
							continue;
						}
#ifdef DEBUG_FIND_EXTREMA_MODE
						std::cout << "find accurate position" << std::endl;
#endif // DEBUG_FIND_EXTREMA_MODE
/*
有问题
赵春江论文中程序是  float scl_octv = kpt.size*0.5 / (1 << o); 
觉得是  1>>o
*/
						//the scale that relative to the current octave 
						float scl_octv = kpt.size*0.5 / (1 >> o); 
						
						//main orientation compute
						float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers + 3) + layer], cv::Point(c1, r1),
							cvRound(SIFT_ORI_RADIUS*scl_octv), SIFT_ORI_SIG_FCTR*scl_octv, hist, n);


#ifdef DEBUG_FIND_EXTREMA_MODE
						std::cout << "calculate histogram" << std::endl;
#endif // DEBUG_FIND_EXTREMA_MODE

						//auxiliary direction threshold
						float mag_thr = float(omax*SIFT_ORI_PEAK_RATIO);
						//compute the feature orientation
						for (int j = 0; j < n; j++)
						{
							//I is the index previous cylinder 
							int I = j > 0 ? j - 1 : n - 1;
							//r2 is the index next cylinder
							int r2 = j < n - 1 ? j + 1 : 0;
							// directional angle fitting
							//the hist[j] value bigger than previous and next hist 
							//meanwhile the hist[j] value bigger than mag_thr
							//then it is the auxiliary direction angle
							if (hist[j] > hist[I] && hist[j] > hist[r2] && hist[j] >= mag_thr)
							{
								float bin = j + 0.5*(hist[I] - hist[r2]) / (hist[I] - 2 * hist[j] + hist[r2]);

								bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
								kpt.angle = 360.0 - (float)((360.0 / n)*bin);

								if (std::abs(kpt.angle - 360.0) < FLT_EPSILON)
									kpt.angle = 0.0;
								keypoints.push_back(kpt);
							}
						}

#ifdef DEBUG_FIND_EXTREMA_MODE
						std::cout << "push kpt" << std::endl;
#endif // DEBUG_FIND_EXTREMA_MODE

					}
				}

			}

		}
	}

}
/*
** description:
   interpolates a scale - space extremum's location and scale to subpixel
   accuracy to form an image feature.rejects features with low contrast
   based on Section 4 of Lowe's paper.
** dog_py:  the DoG pyramid 
** kpt:     the feature point
** octv:    the octave of the feature point 
** layer:   the layer of the octave
** r c:     r is the row coordinate and c is the col coordinate
*/

bool SiftAL::adjustLocalExtrema(const std::vector<cv::Mat>& dog_pyr, cv::KeyPoint& kpt, int octv, int& layer, int& r, int& c, float contrastThreshold, float edgeThreshold)
{
	//img_scale is the coefficient of normalized the image
	const float img_scale = 1.f / (255 * SIFT_FIXPT_SCALE);
	//first-order partial derivative Dx and Dy coefficient  1/(2h)
	const float deriv_scale = img_scale*0.5f;
	//second-order partial derivative Dxx and Dyy coefficient 1/(h^2)
	const float second_deriv_scale = img_scale;
	//second-order partial derivative Dxy coefficient  1/(4h^2)
	const float cross_deriv_scale = img_scale*0.25f;

	float xi = 0, xr = 0, xc = 0, contr;
	int i = 0;
	for (; i < SIFT_MAX_INTERP_STEPS; i++)
	{
		//the index of the DoG pyramid
		int idx = octv*(nOctaveLayers + 2) + layer;

		const cv::Mat& img = dog_pyr[idx];
		const cv::Mat& prev = dog_pyr[idx - 1];
		const cv::Mat& next = dog_pyr[idx + 1];

		//first-order partial derivative

		cv::Vec3f dD(
			(img.at<short>(r, c + 1) - img.at<short>(r, c - 1))*deriv_scale,  //Dx
			(img.at<short>(r + 1, c) - img.at<short>(r - 1, c))*deriv_scale,  //Dy
			(next.at<short>(r, c) - prev.at<short>(r, c - 1))*deriv_scale     //Dsigma
			);
		//second-order partial derivative
		//twice the value of f(i,j) 
		float v2 = (float)img.at<sift_wt>(r, c) * 2;
		float dxx = (img.at<sift_wt>(r, c + 1) + img.at<sift_wt>(r, c - 1) - v2)*second_deriv_scale;
		float dyy = (img.at<sift_wt>(r + 1, c) + img.at<sift_wt>(r - 1, c) - v2)*second_deriv_scale;
		float dss = (next.at<sift_wt>(r, c) + prev.at<sift_wt>(r, c) - v2)*second_deriv_scale;


		float dxy = (img.at<sift_wt>(r + 1, c + 1) + img.at<sift_wt>(r - 1, c - 1) -
			img.at<sift_wt>(r + 1, c - 1) - img.at<sift_wt>(r - 1, c + 1))*cross_deriv_scale;

		float dxs = (next.at<sift_wt>(r, c + 1) + prev.at<sift_wt>(r, c - 1) -
			next.at<sift_wt>(r, c - 1) - prev.at<sift_wt>(r, c + 1))*cross_deriv_scale;

		float dys = (next.at<sift_wt>(r + 1, c) + prev.at<sift_wt>(r - 1, c) -
			next.at<sift_wt>(r - 1, c) - prev.at<sift_wt>(r + 1, c))*cross_deriv_scale;

		//partial derivative matrix H 
		cv::Matx33f H(dxx, dxy, dxs,
			dxy, dyy, dys,
			dxs, dys, dss);
		//function 18---   X''*x = -X'
		cv::Vec3f X = H.solve(dD, cv::DECOMP_LU);
		//add '-'
		xi = -X[2]; //layer coordinate offset
		xr = -X[1]; //row coordinate offset
		xc = -X[0]; //col coordinate offset

		//if all the offset of the three coordinate get by taylor series interpolates is smaller than 0.5,
		//it shows have found feature point
		//then exit the iteration
		if (std::abs(xi) < 0.5&&std::abs(xr) < 0.5&&std::abs(xc) < 0.5)
			break;
		//if anyone of the three coordinate is larger than a big number,
		//it shows this extreme point is not the feature point
		//return false
		if (std::abs(xi) > (float)(INT_MAX / 3) || std::abs(xr) > (float)(INT_MAX / 3) || std::abs(xc) > (float)(INT_MAX / 3))
			return false;

		//if anyone offset is bigger than 0.5, it need to redefine the interpolate center coordinate position
		c += cvRound(xc);
		r += cvRound(xr);
		layer += cvRound(xi);

		//if the coordinates are out of range of pyramid
		//it shows the extreme point is not the feature point
		//return false
		if (layer<1 || layer>nOctaveLayers ||
			c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER ||
			r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER)
			return false;
	}

	//if the number of iterations exceeds SIFT_MAX_INTERP_STEPS
	//it shows the it has not found the accurate position
	if (i >= SIFT_MAX_INTERP_STEPS)
		return false;

	int idx = octv*(nOctaveLayers + 2) + layer;
	const cv::Mat& img = dog_pyr[idx];
	const cv::Mat& prev = dog_pyr[idx - 1];
	const cv::Mat& next = dog_pyr[idx + 1];

	cv::Vec3f dD(
		(img.at<short>(r, c + 1) - img.at<short>(r, c - 1))*deriv_scale,  //Dx
		(img.at<short>(r + 1, c) - img.at<short>(r - 1, c))*deriv_scale,  //Dy
		(next.at<short>(r, c) - prev.at<short>(r, c))*deriv_scale     //Dsigma
	);

	//compute the taylor first-order partial derivative value
	//dot is the point multiply
	float t = dD.dot(cv::Matx31f(xc, xr, xi));

	//1: compute the f(x0) then normalized the f(x0) by multiply img_scale
	//2: add the f(x0)+f'(x0)
	//get the response value
	contr = img.at<sift_wt>(r, c)*img_scale + t*0.5;

	//if the value of the response is lower than contrastThreshold
	//delete the feature point
	//return false
	if (std::abs(contr)*nOctaveLayers < contrastThreshold)
		return false;

	//principal curvature are computed using the trace and det of the Hessian
	float v2 = img.at<sift_wt>(r, c) * 2.0;
	float dxx = (img.at<sift_wt>(r, c + 1) + img.at<sift_wt>(r, c - 1) - v2)*second_deriv_scale;
	float dyy = (img.at<sift_wt>(r + 1, c) + img.at<sift_wt>(r - 1, c) - v2)*second_deriv_scale;
	float dxy = (img.at<sift_wt>(r + 1, c + 1) + img.at<sift_wt>(r - 1, c - 1) -
		img.at<sift_wt>(r + 1, c - 1) - img.at<sift_wt>(r - 1, c + 1))*cross_deriv_scale;
	float tr = dxx + dyy;
	float det = dxx*dyy - dxy*dxy;
	
	if (det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det)
		return false;

	//!!!!!!!!!Note!!!!!Note!!!!!!!!
	//the image input here is double to the image   it includes -1 octave

	//save feature point information
	//x y coordinate
	kpt.pt.x = (c + xc)*(1 << octv);
	kpt.pt.y = (r + xr)*(1 << octv);
	//octave
	kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5) * 255) << 16);
	//scale size
	kpt.size = sigma*powf(2.0, (layer + xi) / nOctaveLayers)*(1 << octv) * 2;
	//f(x0)+f'(x0) value
	kpt.response = std::abs(contr);

	return true;
}
/*
** description:
	computes a gradient orientation histogram at a specified pixel
** params:
	img:    the gaussian scale image where the feature point is located
	pt:     the coordinate of the feature point in the scale image
	radius: the neighbour radius
	sigma:  the variance of the gaussian function
	hist:   the gradient orientation hist
	n:      the bins of the gradient orientation hist
** return value:the main peak of the hist
*/
float SiftAL::calcOrientationHist(const cv::Mat& img, cv::Point pt, int radius, float sigma, float *hist, int n)
{
	//len is the total points around feature point when compute feature point orientation
	int i, j, k, len = (radius * 2 + 1)*(radius * 2 + 1);
	//expf_scale is the const of the exponential of gaussian function
	float expf_scale = -1.0 / (2.0 * sigma*sigma);

	//allocate memory
	cv::AutoBuffer<float> buf(len * 4 + n + 4);
	//X indicate difference in x-axis direction
	//Y indicate difference in y-axis direction
	//Mag indicate the magnitude of gradient
	//Ori indicate the gradient angle
	//W indicate the weight of gauss
	//the above variables in the buf space location:X and Mag share a section of lenght k memory
	//Y and Ori occupied a section of len length memory separately
	// W occupy a section of len+2,
	//the order is :X(Mag), Y, Ori, W
	float *X = buf, *Mag = X, *Y = X + len, *Ori = Y + len, *W = Ori + len;
	//temphist indicate temporary gradient orientation histogram,the space length is n+2
	//space position is on the top of the W
	//the reason why the temphist and W length is n+2,is because need to do circular cycle operation
	//so must leave two position at the begin and end of temphist
	float *temphist = W + len + 2;
	//clear temphist variables
	for (i = 0; i < n; i++)
	{
		temphist[i] = 0.0;
	}
	//compute x-axis y-axis derivative and weight gauss
	for (i = -radius, k = 0; i <= radius; i++)
	{
		//neighbour pixel y-axis coordinate
		int y = pt.y + i;

		if (y <= 0 || y >= img.rows - 1)
			continue;
		for (j = -radius; j <= radius; j++)
		{
			int x = pt.x + j;
			if (x <= 0 || x >= img.cols - 1)
				continue;
			float dx = (float)(img.at<sift_wt>(y, x + 1)) - (float)(img.at<sift_wt>(y, x - 1));
			float dy = (float)(img.at<sift_wt>(y + 1, x)) - (float)(img.at<sift_wt>(y - 1, x));
			X[k] = dx; Y[k] = dy; W[k] = (i*i + j * j)*expf_scale;
			k++;
		}
	}
	//len indicates the actual number of the feature point
	len = k;

	//compute all the pixels' W, Ori, Mag, in the neighbour  
	cv::hal::exp(W, W, len);
	cv::hal::fastAtan2(Y, X, Ori, len, true);
	cv::hal::magnitude(X, Y, Mag, len);
	//compute gradient orientatin histgram
	for (k = 0; k < len; k++)
	{
		//judge the gradient angle belong to which bin
		int bin = cvRound((n / 360.0)*Ori[k]);
		//range limit:use circular cycle make sure which bin it actual belong
		if (bin >= n)
			bin -= n;
		else if (bin < 0)
			bin += n;
		//cumulative the gradient magnitude after deal with the gaussian weight
		temphist[bin] += W[k] * Mag[k];
	}
	//smooth the histgram
	//for the circular cycle,fill in two variables before and after the histgram in advance
	temphist[-1] = temphist[n - 1];
	temphist[-2] = temphist[n - 2];
	temphist[n] = temphist[0];
	temphist[n + 1] = temphist[1];
	for (i = 0; i < n; i++)
	{
		hist[i] = (float)(temphist[i - 2] + temphist[i + 2]) / 16.0f +
			(float)(temphist[i - 1] + temphist[i + 1]) / 4.0f + 
			(float)temphist[i] * 6.0 / 16.0;
	}
	//compute the histgram main peak
	float maxval = hist[0];
	for (i = 1; i < n; i++)
	{
		maxval = std::max(maxval, hist[i]);
	}
	return maxval;
}

void SiftAL::calcDescriptors(const std::vector<cv::Mat>& gpyr, const std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int nOctaveLayers, int firstOctave)
{
	int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;
	//traverse all feature points
	for (size_t i = 0; i < keypoints.size(); i++)
	{
		cv::KeyPoint kpt = keypoints[i];
		int octave = 0xff & kpt.octave;
		int layer = (0xff00 & kpt.octave) >> 8;
		float scale = powf(2.0, -octave);
		//make sure the octave and layer within a reasonable range
		CV_Assert(octave >= firstOctave&&layer <= nOctaveLayers);
		//get current feature point scale of the gaussian pyrmaid relative to base layer it is on
		//size=sigma*2^(s/S)*2
		float size = kpt.size*scale;

		//get current feature point position coordinate at the gaussian scale image
		cv::Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);

		//get the gaussian-scale image of the current feature point
		const cv::Mat& img = gpyr[(octave - firstOctave)*(nOctaveLayers + 3) + layer];

		//get current feature point angle
		float angle = 360.0 - kpt.angle;

		//if the orientation angle is quite near 360 degrees,then set it to 0 degree
		if (std::abs(angle - 360.0) < FLT_EPSILON)
			angle = 0.0;
#ifdef DEBUG_CALC_DESCRIPTORS
		std::cout << "make sure main ori" << std::endl;
#endif // DEBUG_CALC_DESCRIPTORS

		//compute the feature vector of the feature point
		//size*0.5 not sure what is meaning
		calcSIFTDescriptor(img, ptf, angle, size*0.5, d, n, descriptors.ptr<float>((int)i));

#ifdef DEBUG_CALC_DESCRIPTORS
		std::cout << "calculate descri" << std::endl;
#endif // DEBUG_CALC_DESCRIPTORS

	}


	//kpt.pt.x = (c + xc)*(1 << octv);
	//kpt.pt.y = (r + xr)*(1 << octv);
	////octave
	//kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5) * 255) << 16);
	////scale size
	//kpt.size = sigma*powf(2, (layer + xi) / nOctaveLayers)*(1 << octv) * 2;
	////f(x0)+f'(x0) value
	//kpt.response = std::abs(contr);
}

/*
**compute sift feature point feature vector
**scl: sigma*2^(s/S)*2
**d: descriptors width=4
**n: descriptors histgram bins
*/
void SiftAL::calcSIFTDescriptor(const cv::Mat& img, cv::Point2f ptf, float ori, float scl, int d, int n, float *dst)
{
	cv::Point pt(cvRound(ptf.x), cvRound(ptf.y));
	float cos_t = cosf(ori*(float)(CV_PI / 180));
	float sin_t = sin(ori*(float)(CV_PI / 180));
	float bins_per_rad = n / 360.0;
	//constant part of of exponential function of gaussian weight function
	/*
	有问题 d*d/2  感觉应该是-1/(d*d)*0.5
	*/
	float exp_scale = -1.0 / (d*d*2.0);
	float hist_width = SIFT_DESCR_SCL_FCTR*scl;//3*sigma*2^(s/S)*2

	//function 37
	//r=3*σ*(d+1)*sqrt(2)/2
	int radius = cvRound(hist_width*1.4142135623730951*(d + 1)*0.5);

	//clip the radius to the diagonal of the image to avoid autobuffer too large exception
	radius = std::min(radius, (int)sqrtf((double)img.cols*img.cols + img.rows*img.rows));
	//normalization
	cos_t /= hist_width;
	sin_t /= hist_width;

	//len is the number of pixel around the feature point
	//histlen is the number of histgram which is the length of the feature vector.actually,is dxdxn
	//the reason why add 2 to the histlen is because leave some memory space for circular cycle
	int i, j, k;
	int len = (radius * 2 + 1)*(radius * 2 + 1);
	int histlen = (d + 2)*(d + 2)*(n + 2);
	int rows = img.rows, cols = img.cols;

	//build a section of memory space
	cv::AutoBuffer<float> buf(len * 6 + histlen);

	//X indicate x orientation gradient
	//Y indicate y orientation gradient 
	//Mag indicate gradient magnitude
	//Ori indicate gradient angle
	//W indicate weight of gauss
	//among them Y and Mag share a section of memory which length is len
	//the order is:X Y(Mag) Ori W
	float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;

	//here is the variable of the three-dimensional histogram
	//RBin and CBin indicate the coordinate during dxd neighbour range. the length is len
	//hist indicate the histgram value. the length is histlen
	//order is: RBin CBin hist
	float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

	//clear histgram array
	for (i = 0; i < d + 2; i++)
	{
		for (j = 0; j < d + 2; j++)
		{
			for (k = 0; k < n + 2; k++)
			{
				hist[(i*(d + 2) + j)*(n + 2) + k] = 0;
			}
		}
	}
	//traverse the neighbour range of current feature point 
	for (i = -radius, k = 0; i <= radius; i++)
	{
		for (j = -radius; j <= radius; j++)
		{
			//calculate sample's histgram array coords rotated relative to ori.
			//sutract 0.5 so samples that fall e.g. in the center of row 1
			float c_rot = j*cos_t - i*sin_t;
			float r_rot = j*sin_t + i*cos_t;
			float rbin = r_rot + d / 2 - 0.5f;
			float cbin = c_rot + d / 2 - 0.5f;

			//pixel position coordinate
			int r = pt.y + i, c = pt.x + j;

			//make sure that the pixel whether in the square,and whether it beyond the image border
			if (rbin > -1 && rbin<d && cbin>-1 &&
				cbin < d&&r>0 && r < rows - 1 && c>0 && c < cols - 1)
			{
				//compute x and y first-order partial derivate
				//ignore the denominator
				float dx = (float)(img.at<sift_wt>(r, c + 1) - img.at<sift_wt>(r, c - 1));
				float dy = (float)(img.at<sift_wt>(r + 1, c) - img.at<sift_wt>(r - 1, c));

				X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;

				//exp_scale = -1.0 / (d*d*0.5);
				//gaussian weight
/*有疑问*/		
				W[k] = (c_rot*c_rot + r_rot*r_rot)*exp_scale;
				//count the actual number of pixels
				k++;
			}
		}
	}
	len = k;
	cv::hal::fastAtan2(Y, X, Ori, len, true);//gradient angle
	cv::hal::magnitude(X, Y, Mag, len);//gradient magnitude
	cv::hal::exp(W, W, len);//gaussian weight functionn

	//traverse all pixel
	for (k = 0; k < len; k++)
	{
		//get dxd neighbour region coordinate,that the position below the 3D histogram
		float rbin = RBin[k], cbin = CBin[k];
		//get the belong to which SIFT_DESCR_HIST_BINS,which is the 3D histgram height position
		float obin = (Ori[k] - ori)*bins_per_rad;
		//get the gradient magnitude after the gaussian weight 
		float mag = Mag[k] * W[k];
		//round down the integer
		// r0 c0 o0 are the 3D integer part of the coordinate,which indicate it belongs to which square
		int r0 = cvFloor(rbin);
		int c0 = cvFloor(cbin);
		int o0 = cvFloor(obin);
		//decimal part
		//rbin cbin obin are the 3D decimal part of the coordinate,which indicate its position in the square
		rbin -= r0;
		cbin -= c0;
		obin -= o0;

		//if o0 smaller than 0 degree or bigger than 360 degree,then according to the circular cycle
		//adjust the angle to 0-360
		if (o0 < 0)
			o0 += n;
		else if (o0 >= n)
			o0 -= n;

		//histogram update using tri-linear interplation
		float v_r1 = mag*rbin, v_r0 = mag - v_r1;
		float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
		float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
		float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
		float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
		float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
		float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

		//get the pixel index in the 3D histogram
		int idx = ((r0 + 1)*(d + 2) + c0 + 1)*(n + 2) + o0;
		//8 vertices correspond to 8 histogram before move,accmulate them
		hist[idx] += v_rco000;
		hist[idx + 1] += v_rco001;
		hist[idx + (n + 2)] += v_rco010;
		hist[idx + (n + 3)] += v_rco011;
		hist[idx + (d + 2)*(n + 2)] += v_rco100;
		hist[idx + (d + 2)*(n + 2) + 1] += v_rco101;
		hist[idx + (d + 3)*(n + 2)] += v_rco110;
		hist[idx + (d + 3)*(n + 2) + 1] += v_rco111;
	}
	//finalize histgram, since the orientation histograms are circular
	for (i = 0; i < d; i++)
	{
		for (j = 0; j < d; j++)
		{
			int idx = ((i + 1)*(d + 2) + (j + 1))*(n + 2);
			hist[idx] += hist[idx + n];
			hist[idx + 1] += hist[idx + n + 1];
			for (k = 0; k < n; k++)
			{
				dst[(i*d + j)*n + k] = hist[idx + k];
			}
		}
	}
	//copy histogram to the descriptor, apply hysteresis thresholding
	//and scale the result,so that it can be easily converted to byte array
	float nrm2 = 0;
	//feature vector dims
	len = d*d*n;
	for (k = 0; k < len; k++)
	{
		nrm2 += dst[k] * dst[k];
	}
	float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;
	for (i = 0, nrm2 = 0; i < k; i++)
	{
		float val = std::min(dst[i], thr);
		dst[i] = val;
		nrm2 += val*val;
	}

	nrm2 = SIFT_INT_DESCR_FCTR / std::max(std::sqrt(nrm2), FLT_EPSILON);

#if 1
	for (k = 0; k < len; k++)
	{
		//finally normlization feature vector
		dst[k] = cv::saturate_cast<uchar>(dst[k] * nrm2);
	}
#else
	float nrm1 = 0;
	for (k = 0; k < len; k++)
	{
		dst[k] /= nrm2;
		nrm1 += dst[k];
	}
	nrm1 = 1.0 / std::max(nrm1, FLT_EPSILON);
	for (k = 0; k < len; k++)
	{
		dst[k] = std::sqrt(dst[k] * nrm1);
		cv::saturate_cast<uchar>(std::sqrt(dst[k] * nrm1)*SIFT_INT_DESCR_FCTR);
	}
#endif

}

void SiftAL::DescriptorFilter(std::vector<cv::KeyPoint>& keypoints_ori, std::vector<cv::KeyPoint>& keypoints_filter, cv::Mat& descriptors_ori, cv::Mat& descriptors_filter)
{
	int count_many = 0;
	std::vector<int> keyP_index;
	for (int i = 0; i < keypoints_ori.size(); i++)
	{
		count_many = 0;
		float *data = descriptors_ori.ptr<float>(i);
		for (int j = 0; j < 128; j++)
		{
			count_many += data[j];
		}
		if (count_many > 0)
		{
			keyP_index.push_back(i);
		}
	}
	descriptors_filter.create(keyP_index.size(), 128, CV_32F);

	for (int i = 0; i < keyP_index.size(); i++)
	{
		keypoints_filter.push_back(keypoints_ori[keyP_index[i]]);
		float *data_ori = descriptors_ori.ptr<float>(keyP_index[i]);
		float *data_filter = descriptors_filter.ptr<float>(i);

		for (int j = 0; j < 128; j++)
		{
			data_filter[j] = data_ori[j];
		}
	}
}
