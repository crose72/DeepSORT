#include "cmd_line_util.h"
#include "yolov8.h"
#include "reid_engine.hpp"
#include "tracker.hpp"
#include <opencv2/opencv.hpp>

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
    }
}

int main(int argc, char** argv) {
    try {
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
        reidConfig.featureDim = 2048;
        
        // Set input dimensions for ReID model
        Options reidOptions;
        reidOptions.precision = reidConfig.precision;
        reidOptions.maxBatchSize = 1;
        reidOptions.optBatchSize = 1;
        // Set input dimensions - all must be equal since we have fixed input size
        reidOptions.minInputWidth = 128;  // ReID model min input width
        reidOptions.optInputWidth = 128;  // ReID model optimal input width
        reidOptions.maxInputWidth = 128;  // ReID model max input width
        // Set input height
        reidOptions.minInputHeight = 256;  // ReID model min input height
        reidOptions.optInputHeight = 256;  // ReID model optimal input height
        reidOptions.maxInputHeight = 256;  // ReID model max input height
        reidConfig.engineOptions = reidOptions;
        
        ReIDEngine reid(reid_model, "", reidConfig);
        
        // Initialize tracker with default parameters
        float max_cosine_distance = 0.3f;
        int nn_budget = 100;
        int max_age = 30;
        int n_init = 3;
        
        // Create metric with proper MetricType enum
        NearestNeighborDistanceMetric metric(
            NearestNeighborDistanceMetric::MetricType::Cosine,  // Use cosine distance
            max_cosine_distance,  // Matching threshold
            nn_budget  // Budget for gallery features
        );
        
        // Initialize tracker with the metric
        Tracker tracker(metric, max_cosine_distance, max_age, n_init);
        
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
        while (cap.read(frame)) {
            // Detect objects using YOLOv8
            std::vector<Object> detections = yolo.detectObjects(frame);
            
            // Extract features using ReID model
            std::vector<cv::Mat> person_crops;
            for (const auto& det : detections) {
                if (det.label == 0) { // person class
                    cv::Mat crop = frame(det.rect).clone();
                    person_crops.push_back(crop);
                }
            }
            
            std::vector<std::vector<float>> features;
            if (!person_crops.empty()) {
                for (const auto& crop : person_crops) {
                    auto crop_features = reid.extractFeatures(crop);
                    if (!crop_features.empty()) {
                        features.push_back(crop_features[0]);
                    }
                }
            }
            
            // Update tracker
            tracker.predict();
            auto tracking_dets = convertYoloToDeepSORT(detections, features);
            tracker.update(tracking_dets);
            
            // Draw results
            drawTrackingResults(frame, tracker.getTracks());
            
            // Display frame
            cv::imshow("DeepSORT Tracking", frame);
            if (cv::waitKey(1) == 27) break; // ESC to quit
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}