// tracker.hpp
#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>

// Forward declarations
class Detection;
class NearestNeighborDistanceMetric;
class Track;

class Tracker
{
public:
    /**
     * Multi-target tracker.
     *
     * @param metric Distance metric for measurement-to-track association
     * @param max_iou_distance Maximum IOU distance threshold
     * @param max_age Maximum number of missed misses before a track is deleted
     * @param n_init Number of consecutive detections before track confirmation
     */
    Tracker(NearestNeighborDistanceMetric &metric,
            float max_iou_distance = 0.7f,
            int max_age = 30,
            int n_init = 3);

    void predict();
    void cameraUpdate(const std::string &video, int frame);
    void update(const std::vector<Detection> &detections);

private:
    using Match = std::pair<int, int>;

    std::tuple<std::vector<Match>, std::vector<int>, std::vector<int>>
    match(const std::vector<Detection> &detections);

    void initiateTrack(const Detection &detection);

    NearestNeighborDistanceMetric &metric_;
    float max_iou_distance_;
    int max_age_;
    int n_init_;
    std::vector<Track> tracks_;
    int next_id_;
};