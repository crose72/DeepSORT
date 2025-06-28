#include "cmd_line_util.h"
#include "yolov8.h"
#include "reid_engine.hpp"
#include "tracker.hpp"
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

using namespace tracking;
using namespace tracking::matching;

// Helper function to convert YOLOv8 detections to DeepSORT format
std::vector<Detection> convertYoloToDeepSORT(const std::vector<Object>& yoloObjects, 
                                           const std::vector<std::vector<float>>& features) {
    std::vector<Detection> detections;
    for (size_t i = 0; i < yoloObjects.size(); i++) {
        const auto& obj = yoloObjects[i];
        // Only track person class (class 0 in COCO)
        if (obj.label == 0) {
            // Convert rect to TLWH format (top-left, width, height)
            Eigen::Vector4f tlwh;
            tlwh << obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height;
            
            // Convert feature vector to Eigen format
            Eigen::VectorXf feature;
            if (i < features.size()) {
                feature = Eigen::Map<const Eigen::VectorXf>(features[i].data(), features[i].size());
                
                // Validate feature vector
                if (feature.size() != 512) {
                    spdlog::error("Invalid feature dimension in convertYoloToDeepSORT: {}", feature.size());
                    continue;
                }
                
                // Check for NaN/Inf
                if (!feature.allFinite()) {
                    spdlog::error("Feature vector contains NaN/Inf values");
                    continue;
                }
                
                // Check if feature vector is normalized
                float norm = feature.norm();
                if (std::abs(norm - 1.0f) > 1e-4) {
                    spdlog::warn("Feature vector not normalized, norm: {}", norm);
                    feature.normalize();
                }
                
                spdlog::debug("Detection {}: bbox={},{},{},{} conf={:.2f} feature_size={} feature_norm={:.3f}", 
                    i, tlwh[0], tlwh[1], tlwh[2], tlwh[3], obj.probability, feature.size(), feature.norm());
            } else {
                spdlog::warn("No feature vector for detection {}", i);
                continue;  // Skip detections without features
            }
            
            // Create detection with proper constructor
            detections.emplace_back(tlwh, obj.probability, feature);
        }
    }
    return detections;
}

// Helper function to draw tracking results
void drawTrackingResults(cv::Mat& frame, const std::vector<Track>& tracks) {
    for (const auto& track : tracks) {
        if (!track.isConfirmed() || track.getTimeSinceUpdate() > 1) continue;
        
        // Get track state - convert mean state to TLWH format
        const auto& mean = track.getMean();
        cv::Rect_<float> bbox(mean[0], mean[1], mean[2], mean[3]);
        
        // Draw bounding box
        cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
        
        // Draw ID
        std::string id_text = "ID: " + std::to_string(track.getTrackId());
        cv::Point text_pos(bbox.x, bbox.y - 10);
        cv::putText(frame, id_text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        
        // Log track info
        spdlog::debug("Track ID {}: state={},{},{},{} time_since_update={} confirmed={} features={}", 
            track.getTrackId(), mean[0], mean[1], mean[2], mean[3], 
            track.getTimeSinceUpdate(), track.isConfirmed(), track.getFeatures().size());
    }
}

int main(int argc, char** argv) {
    try {
        // Enable debug logging
        spdlog::set_level(spdlog::level::debug);
        
        // Parse command line arguments using our custom parser
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
        
        // Fix batch size to 1
        reidOptions.maxBatchSize = 1;
        reidOptions.optBatchSize = 1;
        
        // Set input dimensions - all must be equal since we have fixed input size
        // For OSNet, we'll set width to 128 and handle height (256) in the preprocessing
        reidOptions.minInputWidth = 128;  // ReID model input width
        reidOptions.optInputWidth = 128;  // ReID model input width
        reidOptions.maxInputWidth = 128;  // ReID model input width
        
        reidConfig.engineOptions = reidOptions;
        
        ReIDEngine reid(reid_model, "", reidConfig);
        
        // Initialize tracker with tuned parameters
        float max_cosine_distance = 0.5f;  // Increased from 0.3 to be more lenient
        int nn_budget = 100;
        int max_age = 60;  // Increased from 30 to be more patient with missing detections
        int n_init = 2;    // Reduced from 3 to confirm tracks faster
        
        // Create metric with proper MetricType enum
        NearestNeighborDistanceMetric metric(
            NearestNeighborDistanceMetric::MetricType::Cosine,  // Use cosine distance
            max_cosine_distance,  // Matching threshold
            nn_budget  // Budget for gallery features
        );
        
        // Initialize tracker with the metric
        Tracker tracker(metric, 0.9f, max_age, n_init);  // Increased IOU threshold from 0.7 to 0.9
        
        // Open video capture
        cv::VideoCapture cap;
        if (input_source.find_first_not_of("0123456789") == std::string::npos) {
            // Input is a number, treat as camera index
            cap.open(std::stoi(input_source));
        } else {
            // Input is a string, treat as video file path
            cap.open(input_source);
        }
        
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video source" << std::endl;
            return -1;
        }
        
        cv::Mat frame;
        int frame_count = 0;
        while (cap.read(frame)) {
            frame_count++;
            spdlog::debug("\n=== Processing frame {} ===", frame_count);
            
            // Detect objects using YOLOv8
            std::vector<Object> detections = yolo.detectObjects(frame);
            spdlog::debug("Found {} detections", detections.size());
            
            // Extract features using ReID model
            std::vector<cv::Mat> person_crops;
            std::vector<std::vector<float>> features;
            int person_count = 0;
            
            for (const auto& det : detections) {
                if (det.label == 0) { // person class
                    person_count++;
                    try {
                        // Log detection info
                        spdlog::debug("Person {} detection: bbox={},{},{},{} conf={:.2f}", 
                            person_count, det.rect.x, det.rect.y, det.rect.width, det.rect.height, det.probability);
                        
                        // Skip low confidence detections
                        if (det.probability < 0.5) {
                            spdlog::debug("Skipping low confidence detection");
                            continue;
                        }
                        
                        // Validate bbox is within frame and has reasonable size
                        if (det.rect.x < 0 || det.rect.y < 0 || 
                            det.rect.x + det.rect.width > frame.cols || 
                            det.rect.y + det.rect.height > frame.rows ||
                            det.rect.width < 10 || det.rect.height < 20) {  // Min size check
                            spdlog::warn("Invalid bbox coordinates or size, skipping");
                            continue;
                        }
                        
                        // Convert CPU Mat to GPU Mat
                        cv::cuda::GpuMat crop_gpu;
                        crop_gpu.upload(frame(det.rect));
                        
                        // Process one person at a time
                        auto crop_features = reid.extractFeatures(crop_gpu);
                        
                        if (!crop_features.empty()) {
                            // Validate feature vector
                            if (crop_features[0].size() != static_cast<size_t>(reidConfig.featureDim)) {
                                spdlog::error("Invalid feature dimension: got {} expected {}", 
                                    crop_features[0].size(), reidConfig.featureDim);
                                continue;
                            }
                            
                            // Check for NaN or Inf values
                            bool has_invalid = false;
                            for (const auto& val : crop_features[0]) {
                                if (std::isnan(val) || std::isinf(val)) {
                                    has_invalid = true;
                                    break;
                                }
                            }
                            if (has_invalid) {
                                spdlog::warn("Feature vector contains NaN or Inf values, skipping");
                                continue;
                            }
                            
                            // Normalize feature vector
                            float norm = 0.0f;
                            for (const auto& val : crop_features[0]) {
                                norm += val * val;
                            }
                            norm = std::sqrt(norm);
                            if (norm > 0.0f) {
                                for (auto& val : crop_features[0]) {
                                    val /= norm;
                                }
                            } else {
                                spdlog::warn("Zero norm feature vector, skipping");
                                continue;
                            }
                            
                            features.push_back(crop_features[0]);
                            
                            // Log feature vector stats
                            float sum = 0.0f, min_val = crop_features[0][0], max_val = crop_features[0][0];
                            for (const auto& val : crop_features[0]) {
                                sum += val;
                                min_val = std::min(min_val, val);
                                max_val = std::max(max_val, val);
                            }
                            float mean = sum / crop_features[0].size();
                            
                            spdlog::debug("Feature vector stats - min: {:.3f}, max: {:.3f}, mean: {:.3f}, norm: {:.3f}", 
                                min_val, max_val, mean, norm);
                        } else {
                            spdlog::warn("Empty features returned for detection");
                        }
                    } catch (const std::exception& e) {
                        spdlog::error("Error processing detection: {}", e.what());
                        continue;  // Skip this detection if there's an error
                    }
                }
            }
            
            // Update tracker
            spdlog::debug("Updating tracker with {} detections and {} feature vectors", 
                person_count, features.size());
            
            try {
                tracker.predict();
                auto tracking_dets = convertYoloToDeepSORT(detections, features);
                spdlog::debug("Created {} tracking detections", tracking_dets.size());
                tracker.update(tracking_dets);
            } catch (const std::exception& e) {
                spdlog::error("Error during tracking update: {}", e.what());
            }
            
            // Draw results
            drawTrackingResults(frame, tracker.getTracks());
            
            // Display frame
            cv::imshow("DeepSORT Tracking", frame);
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