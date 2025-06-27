#include "reid_engine.hpp"
#include <algorithm>

ReIDEngine::ReIDEngine(const std::string &onnxModelPath, const std::string &trtModelPath, const ReIDConfig &config)
    : FEATURE_DIM(config.featureDim) {
    
    // Create the TensorRT engine with the provided options
    m_trtEngine = std::make_unique<Engine<float>>(config.engineOptions);
    
    if (Util::doesFileExist(trtModelPath)) {
        m_trtEngine->loadNetwork(trtModelPath, SUB_VALS, DIV_VALS, NORMALIZE);
    } else {
        m_trtEngine->buildLoadNetwork(onnxModelPath, SUB_VALS, DIV_VALS, NORMALIZE);
    }
}

std::vector<std::vector<float>> ReIDEngine::extractFeatures(const cv::Mat &inputImageBGR) {
    return extractFeatures(inputImageBGR, createZeroTargetFeatures());
}

std::vector<std::vector<float>> ReIDEngine::extractFeatures(const cv::cuda::GpuMat &inputImageBGR) {
    return extractFeatures(inputImageBGR, createZeroTargetFeatures());
}

std::vector<std::vector<float>> ReIDEngine::extractFeatures(const cv::Mat &inputImageBGR, const std::vector<float> &targetFeatures) {
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImageBGR);
    return extractFeatures(gpuImg, targetFeatures);
}

std::vector<std::vector<float>> ReIDEngine::extractFeatures(const cv::cuda::GpuMat &inputImageBGR, const std::vector<float> &targetFeatures) {
    // Preprocess image
    auto preprocessedInputs = preprocess(inputImageBGR);
    
    // Add target features as second input
    std::vector<cv::cuda::GpuMat> targetFeatureChannels;
    cv::cuda::GpuMat targetFeaturesMat(1, FEATURE_DIM, CV_32F);
    targetFeaturesMat.upload(targetFeatures);
    targetFeatureChannels.push_back(targetFeaturesMat);
    
    // Create inputs vector with batch size 1
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;
    inputs.push_back(preprocessedInputs[0]);  // Image input
    inputs.push_back(targetFeatureChannels);  // Target features input
    
    // Prepare output vector
    std::vector<std::vector<std::vector<float>>> outputs;
    
    // Run inference
    m_trtEngine->runInference(inputs, outputs);
    
    // Process outputs
    std::vector<std::vector<float>> features;
    if (!outputs.empty() && !outputs[0].empty()) {
        for (const auto& feature : outputs[0]) {
            // L2 normalize
            float norm = 0.0f;
            for (float val : feature) {
                norm += val * val;
            }
            norm = std::sqrt(norm);
            
            std::vector<float> normalized_feature = feature;
            for (float& val : normalized_feature) {
                val /= norm;
            }
            
            features.push_back(normalized_feature);
        }
    }
    
    return features;
}

std::vector<std::vector<cv::cuda::GpuMat>> ReIDEngine::preprocess(const cv::cuda::GpuMat &input) {
    cv::cuda::GpuMat resized;
    cv::cuda::resize(input, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    
    // Convert to float and normalize to [0,1]
    cv::cuda::GpuMat float_img;
    resized.convertTo(float_img, CV_32F, 1.0/255.0);
    
    // Normalize using ImageNet mean and std
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    
    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(float_img, channels);
    
    for (size_t i = 0; i < channels.size(); i++) {
        channels[i].convertTo(channels[i], CV_32F, 1.0/std.val[i], -mean.val[i]/std.val[i]);
    }
    
    // Return in the format expected by tensorrt-cpp-api: vector<vector<GpuMat>>
    // Outer vector: batch size (1), Inner vector: channels
    std::vector<std::vector<cv::cuda::GpuMat>> preprocessed;
    preprocessed.push_back(channels);
    
    return preprocessed;
} 