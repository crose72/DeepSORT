#include "yolov8.h"
#include "reid_engine.hpp"
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

// Helper function to compute cosine similarity between two feature vectors
float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    return dot / (norm_a * norm_b);
}

int main(int argc, char** argv) {
    try {
        // Enable debug logging
        spdlog::set_level(spdlog::level::debug);
        
        // Parse command line arguments
        if (argc != 4) {
            std::cerr << "Usage: " << argv[0] << " <yolo_model> <reid_model> <input>" << std::endl;
            return -1;
        }
        std::string yolo_model = argv[1];
        std::string reid_model = argv[2];
        std::string input_source = argv[3];
        
        // Initialize YOLOv8
        YoloV8Config yoloConfig;
        yoloConfig.precision = Precision::FP16;
        YoloV8 yolo(yolo_model, "", yoloConfig);
        
        // Initialize ReID
        ReIDConfig reidConfig;
        reidConfig.precision = Precision::FP16;
        reidConfig.featureDim = 512;  // OSNet feature dimension
        
        // Set input dimensions for ReID model
        Options reidOptions;
        reidOptions.precision = reidConfig.precision;
        reidOptions.maxBatchSize = 1;
        reidOptions.optBatchSize = 1;
        reidOptions.minInputWidth = 128;
        reidOptions.optInputWidth = 128;
        reidOptions.maxInputWidth = 128;
        
        reidConfig.engineOptions = reidOptions;
        ReIDEngine reid(reid_model, "", reidConfig);
        
        // Open video capture
        cv::VideoCapture cap;
        if (input_source.find_first_not_of("0123456789") == std::string::npos) {
            cap.open(std::stoi(input_source));
        } else {
            cap.open(input_source);
        }
        
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video source" << std::endl;
            return -1;
        }
        
        // Store last N feature vectors for comparison
        const int feature_history = 5;
        std::vector<std::vector<float>> feature_buffer;
        std::vector<cv::Mat> person_images;  // Store corresponding person crops
        
        cv::Mat frame;
        int frame_count = 0;
        while (cap.read(frame)) {
            frame_count++;
            spdlog::debug("\n=== Processing frame {} ===", frame_count);
            
            // Detect objects using YOLOv8
            std::vector<Object> detections = yolo.detectObjects(frame);
            
            // Process each person detection
            for (const auto& det : detections) {
                if (det.label == 0 && det.probability > 0.5) {  // Person class with good confidence
                    try {
                        // Get person crop
                        cv::Mat person_crop = frame(det.rect).clone();
                        
                        // Extract features
                        cv::cuda::GpuMat crop_gpu;
                        crop_gpu.upload(person_crop);
                        auto features = reid.extractFeatures(crop_gpu);
                        
                        if (!features.empty()) {
                            // Draw detection using YOLOv8's visualization
                            yolo.drawObjectLabels(frame, {det}, 2);
                            
                            // Store feature vector and image if buffer not full
                            if (feature_buffer.size() < feature_history) {
                                feature_buffer.push_back(features[0]);
                                person_images.push_back(person_crop);
                                
                                // Draw text indicating new person
                                cv::putText(frame, "New Person", 
                                    cv::Point(det.rect.x, det.rect.y - 30),  // Moved up to not overlap with YOLO label
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                            } else {
                                // Compare with stored features
                                float max_similarity = 0.0f;
                                int best_match = -1;
                                
                                for (size_t i = 0; i < feature_buffer.size(); i++) {
                                    float sim = cosineSimilarity(features[0], feature_buffer[i]);
                                    if (sim > max_similarity) {
                                        max_similarity = sim;
                                        best_match = i;
                                    }
                                }
                                
                                // Draw similarity score
                                std::string text = cv::format("Sim: %.2f", max_similarity);
                                cv::putText(frame, text,
                                    cv::Point(det.rect.x, det.rect.y - 30),  // Moved up to not overlap with YOLO label
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                                
                                // If very similar to an existing feature, update it
                                if (max_similarity > 0.8) {
                                    feature_buffer[best_match] = features[0];
                                    person_images[best_match] = person_crop;
                                }
                                // If different from all stored features, replace oldest
                                else if (max_similarity < 0.5) {
                                    feature_buffer.erase(feature_buffer.begin());
                                    feature_buffer.push_back(features[0]);
                                    person_images.erase(person_images.begin());
                                    person_images.push_back(person_crop);
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        spdlog::error("Error processing detection: {}", e.what());
                    }
                }
            }
            
            // Show stored person images in a separate window
            if (!person_images.empty()) {
                int grid_width = 128;  // Width of each person crop in grid
                int grid_height = 256; // Height of each person crop in grid
                int cols = std::min(5, (int)person_images.size());
                int rows = (person_images.size() + cols - 1) / cols;
                
                cv::Mat grid(grid_height * rows, grid_width * cols, CV_8UC3, cv::Scalar(0, 0, 0));
                for (size_t i = 0; i < person_images.size(); i++) {
                    int row = i / cols;
                    int col = i % cols;
                    cv::Mat resized;
                    cv::resize(person_images[i], resized, cv::Size(grid_width, grid_height));
                    resized.copyTo(grid(cv::Rect(col * grid_width, row * grid_height, grid_width, grid_height)));
                }
                cv::imshow("Person Gallery", grid);
            }
            
            // Display frame
            cv::imshow("ReID Demo", frame);
            if (cv::waitKey(1) == 27) break; // ESC to quit
            
            // Add a small delay to make the output readable
            cv::waitKey(50);
        }
        
    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        return -1;
    }
    
    return 0;
} 