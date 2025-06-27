#pragma once

#include "engine.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "util/Util.h"

struct ReIDConfig {
    // The precision to be used for inference
    Precision precision = Precision::FP16;
    // Calibration data directory. Must be specified when using INT8 precision.
    std::string calibrationDataDirectory;
    // Feature dimension of the ReID model output
    int featureDim = 2048;
    // Engine options for TensorRT
    Options engineOptions;
};

class ReIDEngine {
public:
    // Builds the onnx model into a TensorRT engine, and loads the engine into memory
    ReIDEngine(const std::string &onnxModelPath, const std::string &trtModelPath, const ReIDConfig &config);

    // Extract features from a batch of person crops
    std::vector<std::vector<float>> extractFeatures(const cv::Mat &inputImageBGR);
    std::vector<std::vector<float>> extractFeatures(const cv::cuda::GpuMat &inputImageBGR);
    
    // Extract features from a batch of person crops with target features
    std::vector<std::vector<float>> extractFeatures(const cv::Mat &inputImageBGR, const std::vector<float> &targetFeatures);
    std::vector<std::vector<float>> extractFeatures(const cv::cuda::GpuMat &inputImageBGR, const std::vector<float> &targetFeatures);
    
private:
    static constexpr int INPUT_WIDTH = 128;
    static constexpr int INPUT_HEIGHT = 256;
    static constexpr int NUM_CHANNELS = 3;
    
    std::unique_ptr<Engine<float>> m_trtEngine;
    
    // Used for image preprocessing - ImageNet normalization
    const std::array<float, 3> SUB_VALS{0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const std::array<float, 3> DIV_VALS{0.229f * 255.f, 0.224f * 255.f, 0.225f * 255.f};
    const bool NORMALIZE = true;
    
    // Feature dimension from config
    int FEATURE_DIM;
    
    // Preprocess the input
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat &input);
    
    // Convert engine outputs to feature vectors and normalize
    std::vector<std::vector<float>> postprocess(std::vector<float> &featureVector);
    
    // Create a zero-initialized target feature vector
    std::vector<float> createZeroTargetFeatures() const {
        return std::vector<float>(FEATURE_DIM, 0.0f);
    }
}; 