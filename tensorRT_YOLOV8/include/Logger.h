#ifndef LOGGER_H
#define LOGGER_H

#include <NvInfer.h>
#include <iostream>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;

class Logger : public nvinfer1::ILogger 
{
public:
    void log(Severity severity, const char *msg) noexcept override 
    {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
};

#endif // LOGGER_H
