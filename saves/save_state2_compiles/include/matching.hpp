#pragma once

#include <Eigen/Dense>
#include <vector>
#include <utility>
#include "track.hpp"
#include "detection.hpp"

namespace tracking {

constexpr float INFTY_COST = 1e+5f;

// Forward declarations
class Track;
class Detection;

namespace matching {

/**
 * @brief Compute intersection over union between bounding boxes
 * 
 * @param bbox Single bounding box
 * @param candidates Matrix of candidate bounding boxes
 * @return Eigen::VectorXf IoU scores
 */
Eigen::VectorXf iou(const Eigen::Vector4f& bbox, const Eigen::MatrixXf& candidates);

/**
 * @brief Compute IoU distance cost matrix between tracks and detections
 */
Eigen::MatrixXf iouCost(const std::vector<Track>& tracks,
                       const std::vector<Detection>& detections,
                       const std::vector<int>& track_indices = {},
                       const std::vector<int>& detection_indices = {});

/**
 * @brief Nearest neighbor distance metric
 */
class NearestNeighborDistanceMetric {
public:
    enum class MetricType {
        Euclidean,
        Cosine
    };

    NearestNeighborDistanceMetric(MetricType metric, float matching_threshold, int budget = -1);

    void partialFit(const std::vector<Eigen::VectorXf>& features,
                    const std::vector<int>& targets,
                    const std::vector<int>& active_targets);

    Eigen::MatrixXf distance(const std::vector<Eigen::VectorXf>& features,
                            const std::vector<int>& targets);

    float getMatchingThreshold() const { return matching_threshold_; }

private:
    MetricType metric_;
    float matching_threshold_;
    int budget_;
    std::map<int, std::vector<Eigen::VectorXf>> samples_;

    float euclideanDistance(const Eigen::VectorXf& x, const Eigen::VectorXf& y) const;
    float cosineDistance(const Eigen::VectorXf& x, const Eigen::VectorXf& y) const;
};

/**
 * @brief Solve linear assignment problem using Hungarian algorithm
 */
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
minCostMatching(const std::function<Eigen::MatrixXf(
                    const std::vector<Track>&,
                    const std::vector<Detection>&,
                    const std::vector<int>&,
                    const std::vector<int>&)>& distance_metric,
                float max_distance,
                const std::vector<Track>& tracks,
                const std::vector<Detection>& detections,
                const std::vector<int>& track_indices = {},
                const std::vector<int>& detection_indices = {});

/**
 * @brief Run matching cascade for track-to-detection association
 */
std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
matchingCascade(const std::function<Eigen::MatrixXf(
                    const std::vector<Track>&,
                    const std::vector<Detection>&,
                    const std::vector<int>&,
                    const std::vector<int>&)>& distance_metric,
                float max_distance,
                int cascade_depth,
                const std::vector<Track>& tracks,
                const std::vector<Detection>& detections,
                const std::vector<int>& track_indices = {},
                const std::vector<int>& detection_indices = {});

/**
 * @brief Gate cost matrix based on Mahalanobis distance
 */
Eigen::MatrixXf gateCostMatrix(const Eigen::MatrixXf& cost_matrix,
                              const std::vector<Track>& tracks,
                              const std::vector<Detection>& detections,
                              const std::vector<int>& track_indices,
                              const std::vector<int>& detection_indices,
                              float gated_cost = INFTY_COST,
                              bool only_position = false);

} // namespace matching
} // namespace tracking 