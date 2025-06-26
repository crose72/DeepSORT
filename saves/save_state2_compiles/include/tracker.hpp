#pragma once

#include <vector>
#include <memory>
#include "track.hpp"
#include "detection.hpp"
#include "matching.hpp"

namespace tracking {

class Tracker {
public:
    /**
     * @brief Construct a new Tracker object
     * 
     * @param metric Distance metric for measurement-to-track association
     * @param max_iou_distance Maximum IOU distance threshold
     * @param max_age Maximum number of missed misses before track deletion
     * @param n_init Number of consecutive detections before track confirmation
     */
    Tracker(matching::NearestNeighborDistanceMetric metric,
            float max_iou_distance = 0.7f,
            int max_age = 30,
            int n_init = 3);

    /**
     * @brief Propagate track state distributions one time step forward
     */
    void predict();

    /**
     * @brief Update camera motion for all tracks
     * 
     * @param video Video identifier
     * @param frame Frame number
     */
    void cameraUpdate(const std::string& video, int frame);

    /**
     * @brief Perform measurement update and track management
     * 
     * @param detections List of detections at the current time step
     */
    void update(const std::vector<Detection>& detections);

    // Getters
    const std::vector<Track>& getTracks() const { return tracks_; }

private:
    /**
     * @brief Create new tracks for unmatched detections
     * 
     * @param detection Detection to create track from
     */
    void initiateTrack(const Detection& detection);

    /**
     * @brief Match tracks and detections
     * 
     * @param detections List of detections
     * @return std::tuple<matches, unmatched_tracks, unmatched_detections>
     */
    std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
    match(const std::vector<Detection>& detections);

private:
    // Track management
    std::vector<Track> tracks_;
    int next_id_;

    // Track parameters
    matching::NearestNeighborDistanceMetric metric_;
    float max_iou_distance_;
    int max_age_;
    int n_init_;
};

} // namespace tracking 