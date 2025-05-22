#include "InferenceEngine.h"
#include <opencv2/opencv.hpp>
#include <iostream>

InferenceEngine::InferenceEngine(CameraCapture &camera, TensorRTModel &model) : camera(camera), model(model) {}

void InferenceEngine::run()
{
    int inputSize = 1 * 3 * 224 * 224;
    int outputSize = 1 * 1000; // Adjust based on your model output size
    float *inputData = new float[inputSize];
    float *outputData = new float[outputSize];
    void *inputDevice = nullptr;
    void *outputDevice = nullptr;
    cudaMalloc(&inputDevice, inputSize * sizeof(float));
    cudaMalloc(&outputDevice, outputSize * sizeof(float));

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int frameCount = 0; // Track the frame number

    while (true)
    {
        cv::Mat frame;
        if (!camera.getFrame(frame))
        {
            break;
        }

        // Preprocess the frame
        cv::Mat image = loadImageFromFrame(frame, 224, 224);
        prepareInputData(image, inputData);

        // Copy input data to GPU
        cudaMemcpyAsync(inputDevice, inputData, inputSize * sizeof(float), cudaMemcpyHostToDevice, stream);

        // Run inference
        model.doInference(reinterpret_cast<float *>(inputDevice), reinterpret_cast<float *>(outputDevice), inputSize, outputSize, stream);

        // Copy output data from GPU to host
        cudaMemcpyAsync(outputData, outputDevice, outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Display output (For example, print first 5 results)
        for (int i = 0; i < 20; i++)
        {
            std::cout << "Output[" << i << "] = " << outputData[i] << std::endl;
        }

        // Display the frame with the results
        std::string frameText = "Frame: " + std::to_string(frameCount);
        cv::putText(frame, frameText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        // Show the frame with the output text
        cv::imshow("Webcam Feed", frame);

        // Exit if 'q' is pressed
        if (cv::waitKey(1) == 'q')
        {
            break;
        }

        frameCount++; // Increment the frame number
    }

    // Cleanup
    delete[] inputData;
    delete[] outputData;
    cudaFree(inputDevice);
    cudaFree(outputDevice);
    cudaStreamDestroy(stream);
}

cv::Mat InferenceEngine::loadImageFromFrame(cv::Mat &frame, int inputHeight, int inputWidth)
{
    // Resize image
    cv::Mat resizedImage;
    cv::resize(frame, resizedImage, cv::Size(inputWidth, inputHeight));

    // Convert to float and normalize
    resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255.0);

    // Convert from BGR to RGB
    cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);

    return resizedImage;
}

void InferenceEngine::prepareInputData(const cv::Mat &image, float *inputData)
{
    int height = image.rows;
    int width = image.cols;
    int channels = image.channels();

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int c = 0; c < channels; c++)
            {
                inputData[c * height * width + i * width + j] = image.at<cv::Vec3f>(i, j)[c];
            }
        }
    }
}
