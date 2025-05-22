#ifndef MYCUSTOMPLUGIN_H
#define MYCUSTOMPLUGIN_H

#include <NvInfer.h>
#include <iostream>

using namespace nvinfer1;

class MyCustomPlugin : public IPluginV2 {
public:
    MyCustomPlugin() = default;

    // 获取输出张量数量
    int getNbOutputs() const override {
        return 1;  // 输出一个张量
    }

    // 返回输出张量的维度
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
        // 假设输入张量的维度是相同的，所以直接返回第一个输入张量的维度
        return inputs[0];
    }

    // 配置插件的输入输出格式
    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims,
                             int nbOutputs, DataType type, PluginFormat format) override {
        // 这里我们不需要做额外的配置，保持默认设置
    }

    // 获取插件的工作空间大小（这里我们不需要额外的工作空间）
    size_t getWorkspaceSize(int) const override {
        return 0;
    }

    // 插件的核心计算逻辑
    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace,
                cudaStream_t stream) override {
        // 假设每个输入张量的元素类型是 float，且输入张量的形状已通过 getOutputDimensions 确定
        const float* input1 = static_cast<const float*>(inputs[0]);
        const float* input2 = static_cast<const float*>(inputs[1]);
        float* output = static_cast<float*>(outputs[0]);

        // 简单的加法操作：将两个输入张量相加并将结果写入输出张量
        int numElements = batchSize * volume(inputs[0]);
        cudaMemcpyAsync(output, input1, numElements * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaAdd<<<(numElements + 255) / 256, 256, 0, stream>>>(output, input2, numElements);

        return 0;
    }

    // 返回插件类型名称
    const char* getPluginType() const override {
        return "MyCustomPlugin";
    }

    // 返回插件版本
    const char* getPluginVersion() const override {
        return "1.0";
    }

    // 克隆插件实例
    IPluginV2* clone() const override {
        return new MyCustomPlugin(*this);
    }

    // 销毁插件实例
    void destroy() override {
        delete this;
    }

    // 初始化插件
    void initialize() override {}

    // 终止插件
    void terminate() override {}

    // 获取插件命名空间
    void setPluginNamespace(const char* libNamespace) override {
        mNamespace = libNamespace;
    }

    // 获取插件命名空间
    const char* getPluginNamespace() const override {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace;

    // 计算张量的元素数量
    int volume(const Dims& dims) const {
        int vol = 1;
        for (int i = 0; i < dims.nbDims; i++) {
            vol *= dims.d[i];
        }
        return vol;
    }
};

#endif // MYCUSTOMPLUGIN_H
