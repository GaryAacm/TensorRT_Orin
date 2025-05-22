#include "camera.h"
#include "tensorRTModel.h"
#include "InferenceEngine.h"
#include "Logger.h"

int main() 
{
    Logger logger;
    std::string onnxModelPath = "./model.onnx";
    std::string engineFilePath = "./model.engine";

    try 
    {
        CameraCapture camera;
        TensorRTModel model(onnxModelPath, engineFilePath, logger);
        InferenceEngine inferenceEngine(camera, model);
        inferenceEngine.run();
    } catch (const std::exception &e) 
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
