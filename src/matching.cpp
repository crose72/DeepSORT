#include "matching.hpp"
#include "munkres.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <opencv2/core/eigen.hpp>

namespace tracking
{
    namespace matching
    {

        Eigen::VectorXf iou(const Eigen::Vector4f &bbox, const Eigen::MatrixXf &candidates)
        {
            Eigen::Vector2f bbox_tl = bbox.head(2);
            Eigen::Vector2f bbox_br = bbox.head(2) + bbox.tail(2);

            Eigen::MatrixXf candidates_tl = candidates.leftCols(2);
            Eigen::MatrixXf candidates_br = candidates.leftCols(2) + candidates.rightCols(2);

            int num_candidates = candidates.rows();
            Eigen::VectorXf iou_scores(num_candidates);

            for (int i = 0; i < num_candidates; ++i)
            {
                Eigen::Vector2f tl = candidates_tl.row(i);
                Eigen::Vector2f br = candidates_br.row(i);

                // Calculate intersection
                Eigen::Vector2f intersection_tl = tl.cwiseMax(bbox_tl);
                Eigen::Vector2f intersection_br = br.cwiseMin(bbox_br);
                Eigen::Vector2f wh = (intersection_br - intersection_tl).cwiseMax(0.0f);
                float area_intersection = wh.prod();

                // Calculate union
                float area_bbox = bbox.tail(2).prod();
                float area_candidate = candidates.row(i).tail(2).prod();
                float area_union = area_bbox + area_candidate - area_intersection;

                iou_scores(i) = area_intersection / area_union;
            }

            return iou_scores;
        }

        Eigen::MatrixXf iouCost(const std::vector<Track> &tracks,
                                const std::vector<Detection> &detections,
                                const std::vector<int> &track_indices,
                                const std::vector<int> &detection_indices)
        {

            std::vector<int> _track_indices = track_indices;
            if (_track_indices.empty())
            {
                _track_indices.resize(tracks.size());
                std::iota(_track_indices.begin(), _track_indices.end(), 0);
            }

            std::vector<int> _detection_indices = detection_indices;
            if (_detection_indices.empty())
            {
                _detection_indices.resize(detections.size());
                std::iota(_detection_indices.begin(), _detection_indices.end(), 0);
            }

            Eigen::MatrixXf cost_matrix = Eigen::MatrixXf::Zero(_track_indices.size(), _detection_indices.size());

            for (size_t i = 0; i < _track_indices.size(); ++i)
            {
                int track_idx = _track_indices[i];
                if (tracks[track_idx].getTimeSinceUpdate() > 1)
                {
                    cost_matrix.row(i).setConstant(INFTY_COST);
                    continue;
                }

                Eigen::Vector4f bbox = tracks[track_idx].toTLWH();
                Eigen::MatrixXf candidates(_detection_indices.size(), 4);
                for (size_t j = 0; j < _detection_indices.size(); ++j)
                {
                    candidates.row(j) = detections[_detection_indices[j]].getTLWH();
                }

                Eigen::VectorXf iou_scores = iou(bbox, candidates);
                cost_matrix.row(i) = Eigen::RowVectorXf::Constant(_detection_indices.size(), 1.0f) - iou_scores.transpose();
            }

            return cost_matrix;
        }

        NearestNeighborDistanceMetric::NearestNeighborDistanceMetric(
            MetricType metric, float matching_threshold, int budget)
            : metric_(metric), matching_threshold_(matching_threshold), budget_(budget)
        {
        }

        void NearestNeighborDistanceMetric::partialFit(
            const std::vector<Eigen::VectorXf> &features,
            const std::vector<int> &targets,
            const std::vector<int> &active_targets)
        {

            for (size_t i = 0; i < features.size(); ++i)
            {
                int target = targets[i];
                samples_[target].push_back(features[i]);

                if (budget_ > 0)
                {
                    if (samples_[target].size() > static_cast<size_t>(budget_))
                    {
                        samples_[target].erase(samples_[target].begin());
                    }
                }
            }

            // Keep only active targets
            std::map<int, std::vector<Eigen::VectorXf>> active_samples;
            for (int target : active_targets)
            {
                if (samples_.find(target) != samples_.end())
                {
                    active_samples[target] = samples_[target];
                }
            }
            samples_ = std::move(active_samples);
        }

        Eigen::MatrixXf NearestNeighborDistanceMetric::distance(
            const std::vector<Eigen::VectorXf> &features,
            const std::vector<int> &targets)
        {

            Eigen::MatrixXf cost_matrix(targets.size(), features.size());

            for (size_t i = 0; i < targets.size(); ++i)
            {
                const auto &target_features = samples_[targets[i]];
                for (size_t j = 0; j < features.size(); ++j)
                {
                    float min_cost = std::numeric_limits<float>::infinity();
                    for (const auto &target_feature : target_features)
                    {
                        float cost = (metric_ == MetricType::Euclidean) ? euclideanDistance(target_feature, features[j]) : cosineDistance(target_feature, features[j]);
                        min_cost = std::min(min_cost, cost);
                    }
                    cost_matrix(i, j) = min_cost;
                }
            }

            return cost_matrix;
        }

        float NearestNeighborDistanceMetric::euclideanDistance(
            const Eigen::VectorXf &x, const Eigen::VectorXf &y) const
        {
            return (x - y).squaredNorm();
        }

        float NearestNeighborDistanceMetric::cosineDistance(
            const Eigen::VectorXf &x, const Eigen::VectorXf &y) const
        {
            return 1.0f - x.normalized().dot(y.normalized());
        }

        std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
        minCostMatching(const std::function<Eigen::MatrixXf(
                            const std::vector<Track> &,
                            const std::vector<Detection> &,
                            const std::vector<int> &,
                            const std::vector<int> &)> &distance_metric,
                        float max_distance,
                        const std::vector<Track> &tracks,
                        const std::vector<Detection> &detections,
                        const std::vector<int> &track_indices,
                        const std::vector<int> &detection_indices)
        {

            std::vector<int> _track_indices = track_indices;
            std::vector<int> _detection_indices = detection_indices;

            if (_track_indices.empty())
            {
                _track_indices.resize(tracks.size());
                std::iota(_track_indices.begin(), _track_indices.end(), 0);
            }
            if (_detection_indices.empty())
            {
                _detection_indices.resize(detections.size());
                std::iota(_detection_indices.begin(), _detection_indices.end(), 0);
            }

            if (_detection_indices.empty() || _track_indices.empty())
            {
                return {{}, _track_indices, _detection_indices};
            }

            Eigen::MatrixXf cost_matrix = distance_metric(tracks, detections, _track_indices, _detection_indices);
            cost_matrix = (cost_matrix.array() > max_distance).select(max_distance + 1e-5f, cost_matrix);

            // Convert to OpenCV matrix for Munkres algorithm
            cv::Mat cost_matrix_cv(cost_matrix.rows(), cost_matrix.cols(), CV_32F);
            cv::eigen2cv(cost_matrix, cost_matrix_cv);

            // Use Munkres algorithm for optimal assignment
            Munkres munkres;
            auto assignments = munkres.compute(cost_matrix_cv);

            std::vector<std::pair<int, int>> matches;
            std::vector<int> unmatched_tracks = _track_indices;
            std::vector<int> unmatched_detections = _detection_indices;

            for (const auto& assignment : assignments) {
                int i = assignment.first;
                int j = assignment.second;
                if (cost_matrix(i, j) < max_distance) {
                    matches.emplace_back(_track_indices[i], _detection_indices[j]);
                    unmatched_tracks.erase(std::remove(unmatched_tracks.begin(),
                                                       unmatched_tracks.end(),
                                                       _track_indices[i]),
                                           unmatched_tracks.end());
                    unmatched_detections.erase(std::remove(unmatched_detections.begin(),
                                                           unmatched_detections.end(),
                                                           _detection_indices[j]),
                                               unmatched_detections.end());
                }
            }

            return std::make_tuple(matches, unmatched_tracks, unmatched_detections);
        }

        std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
        matchingCascade(const std::function<Eigen::MatrixXf(
                            const std::vector<Track> &,
                            const std::vector<Detection> &,
                            const std::vector<int> &,
                            const std::vector<int> &)> &distance_metric,
                        float max_distance,
                        int cascade_depth,
                        const std::vector<Track> &tracks,
                        const std::vector<Detection> &detections,
                        const std::vector<int> &track_indices,
                        const std::vector<int> &detection_indices)
        {

            std::vector<int> _track_indices = track_indices;
            std::vector<int> _detection_indices = detection_indices;

            if (_track_indices.empty())
            {
                _track_indices.resize(tracks.size());
                std::iota(_track_indices.begin(), _track_indices.end(), 0);
            }
            if (_detection_indices.empty())
            {
                _detection_indices.resize(detections.size());
                std::iota(_detection_indices.begin(), _detection_indices.end(), 0);
            }

            std::vector<std::pair<int, int>> matches;
            std::vector<int> unmatched_detections = _detection_indices;

            // Matching cascade
            for (int level = 0; level < cascade_depth; ++level)
            {
                if (unmatched_detections.empty())
                    break;

                std::vector<int> track_indices_l;
                for (int track_idx : _track_indices)
                {
                    if (tracks[track_idx].getTimeSinceUpdate() == 1 + level)
                    {
                        track_indices_l.push_back(track_idx);
                    }
                }

                if (track_indices_l.empty())
                    continue;

                auto [matches_l, _, unmatched_detections_l] = minCostMatching(
                    distance_metric, max_distance, tracks, detections,
                    track_indices_l, unmatched_detections);

                unmatched_detections = unmatched_detections_l;
                matches.insert(matches.end(), matches_l.begin(), matches_l.end());
            }

            std::vector<int> unmatched_tracks;
            for (int track_idx : _track_indices)
            {
                bool is_matched = false;
                for (const auto &match : matches)
                {
                    if (match.first == track_idx)
                    {
                        is_matched = true;
                        break;
                    }
                }
                if (!is_matched)
                {
                    unmatched_tracks.push_back(track_idx);
                }
            }

            return {matches, unmatched_tracks, unmatched_detections};
        }

        Eigen::MatrixXf gateCostMatrix(
            const Eigen::MatrixXf &cost_matrix,
            const std::vector<Track> &tracks,
            const std::vector<Detection> &detections,
            const std::vector<int> &track_indices,
            const std::vector<int> &detection_indices,
            float gated_cost,
            bool only_position)
        {

            float gating_threshold = KalmanFilter::chi2inv95.at(4);
            Eigen::MatrixXf measurements(detection_indices.size(), 4);

            for (size_t i = 0; i < detection_indices.size(); ++i)
            {
                measurements.row(i) = detections[detection_indices[i]].toXYAH();
            }

            Eigen::MatrixXf gated_cost_matrix = cost_matrix;

            for (size_t i = 0; i < track_indices.size(); ++i)
            {
                const Track &track = tracks[track_indices[i]];
                Eigen::VectorXf gating_distance = track.getKalmanFilter().gatingDistance(
                    track.getMean(), track.getCovariance(), measurements, only_position);

                gated_cost_matrix.row(i) = (gating_distance.array() > gating_threshold)
                                               .select(gated_cost, gated_cost_matrix.row(i));
            }

            return gated_cost_matrix;
        }

    } // namespace matching
} // namespace tracking