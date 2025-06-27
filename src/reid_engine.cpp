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
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImageBGR);
    return extractFeatures(gpuImg);
}

std::vector<std::vector<float>> ReIDEngine::extractFeatures(const cv::cuda::GpuMat &inputImageBGR) {
    try {
        // Preprocess image
        auto preprocessedInputs = preprocess(inputImageBGR);
        if (preprocessedInputs.empty() || preprocessedInputs[0].empty()) {
            spdlog::error("Failed to preprocess input image");
            return {};
        }
        
        // Prepare output vector
        std::vector<std::vector<std::vector<float>>> outputs;
        
        // Run inference
        if (!m_trtEngine->runInference(preprocessedInputs, outputs)) {
            spdlog::error("Failed to run inference");
            return {};
        }
        
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
                
                if (norm > 0.0f) {  // Avoid division by zero
                    std::vector<float> normalized_feature = feature;
                    for (float& val : normalized_feature) {
                        val /= norm;
                    }
                    features.push_back(normalized_feature);
                }
            }
        }
        
        return features;
    } catch (const std::exception& e) {
        spdlog::error("Error in extractFeatures: {}", e.what());
        return {};
    }
}

std::vector<std::vector<cv::cuda::GpuMat>> ReIDEngine::preprocess(const cv::cuda::GpuMat &input) {
    try {
        // Resize input to expected dimensions
        cv::cuda::GpuMat resized;
        cv::cuda::resize(input, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));

        // Convert to float and normalize to [0,1]
        cv::cuda::GpuMat float_img;
        resized.convertTo(float_img, CV_32F, 1.0/255.0);

        // Apply normalization (per channel)
        std::vector<cv::cuda::GpuMat> channels;
        cv::cuda::split(float_img, channels);
        for (int i = 0; i < NUM_CHANNELS; i++) {
            channels[i].convertTo(channels[i], CV_32F, 1.0/DIV_VALS[i], -SUB_VALS[i]/DIV_VALS[i]);
        }
        cv::cuda::GpuMat normalized;
        cv::cuda::merge(channels, normalized);

        // Now, normalized is a single 3-channel image
        // Batch size 1: inner vector has 1 image
        std::vector<std::vector<cv::cuda::GpuMat>> preprocessed;
        preprocessed.push_back({normalized});
        return preprocessed;
    } catch (const std::exception& e) {
        spdlog::error("Error in preprocess: {}", e.what());
        return {};
    }
} 