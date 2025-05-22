#include "tensorRTModel.h"
#include <iostream>
#include <fstream>

TensorRTModel::TensorRTModel(const std::string &onnxModelPath, const std::string &engineFilePath, Logger &logger)
    : logger(logger)
{
    engine = loadEngine(engineFilePath);
    if (!engine)
    {
        std::cerr << "Failed to load engine, trying to create a new engine from ONNX" << std::endl;
        engine = createEngineFromOnnx(onnxModelPath);
        if (!engine)
        {
            std::cerr << "Failed to create engine from ONNX" << std::endl;
            throw std::runtime_error("Failed to create or load engine");
        }
        saveEngine(engineFilePath);
    }
    context = engine->createExecutionContext();
}

TensorRTModel::~TensorRTModel()
{
    if (context)
        delete context;
    if (engine)
        delete engine;
}

void TensorRTModel::doInference(float *inputData, float *outputData, int inputSize, int outputSize, cudaStream_t stream)
{
    void *buffers[2] = {inputData, outputData};
    context->executeV2(buffers);
    cudaStreamSynchronize(stream);
}

ICudaEngine *TensorRTModel::createEngineFromOnnx(const std::string &onnxModelPath)
{
    IBuilder *builder = createInferBuilder(logger);
    uint32_t flag = 0;
    INetworkDefinition *network = builder->createNetworkV2(flag);
    IParser *parser = createParser(*network, logger);

    std::ifstream modelFile(onnxModelPath, std::ios::binary);
    if (!modelFile)
    {
        std::cerr << "Unable to open model file: " << onnxModelPath << std::endl;
        return nullptr;
    }
    modelFile.seekg(0, std::ios::end);
    size_t modelSize = modelFile.tellg();
    modelFile.seekg(0, std::ios::beg);
    std::vector<char> modelData(modelSize);
    modelFile.read(modelData.data(), modelSize);
    modelFile.close();

    if (!parser->parse(modelData.data(), modelSize))
    {
        std::cerr << "Failed to parse ONNX model" << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cerr << parser->getError(i)->desc() << std::endl;
        }
        return nullptr;
    }

    IBuilderConfig *config = builder->createBuilderConfig();
    if (builder->platformHasFastFp16())
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    IHostMemory *serializedModel = builder->buildSerializedNetwork(*network, *config);
    if (!serializedModel)
    {
        std::cerr << "Failed to build serialized network" << std::endl;
        return nullptr;
    }

    IRuntime *runtime = createInferRuntime(logger);
    return runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());
}

ICudaEngine *TensorRTModel::loadEngine(const std::string &engineFilePath)
{
    std::ifstream engineFile(engineFilePath, std::ios::binary);
    if (!engineFile)
    {
        return nullptr;
    }
    std::vector<char> engineData((std::istreambuf_iterator<char>(engineFile)),
                                 std::istreambuf_iterator<char>());
    engineFile.close();

    IRuntime *runtime = createInferRuntime(logger);
    return runtime->deserializeCudaEngine(engineData.data(), engineData.size());
}

void TensorRTModel::saveEngine(const std::string &engineFilePath)
{
    IHostMemory *serializedModel = engine->serialize();
    std::ofstream engineFile(engineFilePath, std::ios::binary);
    engineFile.write(reinterpret_cast<const char *>(serializedModel->data()), serializedModel->size());
    engineFile.close();
}
