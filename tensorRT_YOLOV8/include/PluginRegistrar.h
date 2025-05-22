#ifndef PLUGINREGISTRAR_H
#define PLUGINREGISTRAR_H

#include <NvInferPlugin.h>
#include "MyCustomPlugin.h"

class PluginRegistrar 
{
public:
    static void registerPlugins() 
    {
        static bool registered = false;
        if (!registered) 
        {
            // 注册自定义插件
            auto pluginFactory = getPluginRegistry()->getPluginCreator("MyCustomPlugin", "1.0");
            getPluginRegistry()->registerCreator(*pluginFactory);
            registered = true;
        }
    }
};

#endif // PLUGINREGISTRAR_H

