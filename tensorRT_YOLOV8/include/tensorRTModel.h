#ifndef TENSORRTMODEL_H
#define TENSORRTMODEL_H

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <string>
#include <vector>
#include "Logger.h"

using namespace nvinfer1;
using namespace nvonnxparser;

class TensorRTModel 
{
public:
    TensorRTModel(const std::string &onnxModelPath, const std::string &engineFilePath, Logger &logger);
    ~TensorRTModel();
    void doInference(float *inputData, float *outputData, int inputSize, int outputSize, cudaStream_t stream);

private:
    Logger &logger;
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;

    ICudaEngine* createEngineFromOnnx(const std::string &onnxModelPath);
    ICudaEngine* loadEngine(const std::string &engineFilePath);
    void saveEngine(const std::string &engineFilePath);
};

#endif // TENSORRTMODEL_H
