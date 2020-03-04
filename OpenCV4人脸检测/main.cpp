#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::dnn;

#include <iostream>
#include <cstdlib>
using namespace std;
/*
 * opencv4.1.0 need c++11
 * 参考链接
 https://mp.weixin.qq.com/s?__biz=MzA4MDExMDEyMw==&mid=2247487898&idx=1&sn=5997c2fa79e5f20a56214d3b1a7e4e5d&chksm=9fa866dea8dfefc87e5d57da04e8a7d640f7d96f8cc4b7861024e1f5da4a4bf5724e892cfbaf&mpshare=1&scene=1&srcid=&sharer_sharetime=1581471697069&sharer_shareid=a95c16f137e9c84a09b155dc03ebd8da&exportkey=AZudr97SJsK9OMk%2BhdeHyqE%3D&pass_ticket=4JDs8qnXsRe3LnlorrYnXuKQThniJg2trxOipnyjxrP1bL5Tc4nwnIoot%2FcBEmnj#rd
 */
const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);
const float confidenceThreshold = 0.7;
void face_detect_dnn();
void mtcnn_demo();
int main(int argc, char** argv)
{
    face_detect_dnn();
    waitKey(0);
    return 0;
}

void face_detect_dnn() {
    //String modelDesc = "D:/projects/opencv_tutorial/data/models/resnet/deploy.prototxt";
    // String modelBinary = "D:/projects/opencv_tutorial/data/models/resnet/res10_300x300_ssd_iter_140000.caffemodel";
    String modelBinary = "../model/opencv_face_detector_uint8.pb";
    String modelDesc = "../model/opencv_face_detector.pbtxt";
    // 初始化网络
    // dnn::Net net = readNetFromCaffe(modelDesc, modelBinary);
    dnn::Net net = readNetFromTensorflow(modelBinary, modelDesc);

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    if (net.empty())
    {
        printf("could not load net...\n");
        return;
    }

    // 打开摄像头
    // VideoCapture capture(0);
    VideoCapture capture(0);
    if (!capture.isOpened()) {
        printf("could not load camera...\n");
        return;
    }

    Mat frame;
    int count=0;
    while (capture.read(frame)) {
        int64 start = getTickCount();
        if (frame.empty())
        {
            break;
        }
        // 水平镜像调整
        // flip(frame, frame, 1);
        imshow("input", frame);
        if (frame.channels() == 4)
            cvtColor(frame, frame, COLOR_BGRA2BGR);

        // 输入数据调整
        Mat inputBlob = blobFromImage(frame, inScaleFactor,
            Size(inWidth, inHeight), meanVal, false, false);
        net.setInput(inputBlob, "data");

        // 人脸检测
        Mat detection = net.forward("detection_out");
        vector<double> layersTimings;
        double freq = getTickFrequency() / 1000;
        double time = net.getPerfProfile(layersTimings) / freq;
        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        ostringstream ss;
        for (int i = 0; i < detectionMat.rows; i++)
        {
            // 置信度 0～1之间
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > confidenceThreshold)
            {
                count++;
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                Rect object((int)xLeftBottom, (int)yLeftBottom,
                    (int)(xRightTop - xLeftBottom),
                    (int)(yRightTop - yLeftBottom));

                rectangle(frame, object, Scalar(0, 255, 0));

                ss << confidence;
                String conf(ss.str());
                String label = "Face: " + conf;
                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
                    Size(labelSize.width, labelSize.height + baseLine)),
                    Scalar(255, 255, 255), FILLED);
                putText(frame, label, Point(xLeftBottom, yLeftBottom),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
            }
        }
        float fps = getTickFrequency() / (getTickCount() - start);
        ss.str("");
        ss << "FPS: " << fps << " ; inference time: " << time << " ms";
        putText(frame, ss.str(), Point(20, 20), 0, 0.75, Scalar(0, 0, 255), 2, 8);
        imshow("dnn_face_detection", frame);
        if (waitKey(1) >= 0) break;
    }
    printf("total face: %d\n", count);
}
