#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>
#include <stdexcept>

class CameraCapture 
{
public:
    CameraCapture(int device = 0);
    ~CameraCapture();
    bool getFrame(cv::Mat &frame);

private:
    cv::VideoCapture cap;
};

#endif // CAMERACAPTURE_H
