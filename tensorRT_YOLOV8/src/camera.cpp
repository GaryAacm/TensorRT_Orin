#include "camera.h"
#include <iostream>

CameraCapture::CameraCapture(int device) 
{
    cap.open(device);
    if (!cap.isOpened()) 
    {
        std::cerr << "Unable to open camera" << std::endl;
        throw std::runtime_error("Unable to open camera");
    }
}

CameraCapture::~CameraCapture() 
{
    cap.release();
}

bool CameraCapture::getFrame(cv::Mat &frame) 
{
    cap >> frame;
    return !frame.empty();
}
