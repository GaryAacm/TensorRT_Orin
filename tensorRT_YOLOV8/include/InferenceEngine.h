#ifndef INFERENCEENGINE_H
#define INFERENCEENGINE_H

#include <opencv2/opencv.hpp>
#include "camera.h"
#include "tensorRTModel.h"

class InferenceEngine 
{
public:
    InferenceEngine(CameraCapture &camera, TensorRTModel &model);
    void run();

private:
    CameraCapture &camera;
    TensorRTModel &model;

    cv::Mat loadImageFromFrame(cv::Mat &frame, int inputHeight, int inputWidth);
    void prepareInputData(const cv::Mat &image, float *inputData);
};

#endif // INFERENCEENGINE_H
