#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "kalman_filter.hpp"
#include "detection.hpp"

namespace tracking {

enum class TrackState {
    Tentative = 1,
    Confirmed = 2,
    Deleted = 3
};

class Track {
public:
    /**
     * @brief Construct a new Track object
     * 
     * @param detection Initial detection
     * @param track_id Unique track identifier
     * @param n_init Number of consecutive detections before track confirmation
     * @param max_age Maximum number of consecutive misses before track deletion
     */
    Track(const Detection& detection, int track_id, int n_init, int max_age);

    /**
     * @brief Get current position in bounding box format (top left x, top left y, width, height)
     */
    Eigen::Vector4f toTLWH() const;

    /**
     * @brief Get current position in bounding box format (min x, min y, max x, max y)
     */
    Eigen::Vector4f toTLBR() const;

    /**
     * @brief Propagate the state distribution to the current time step using Kalman prediction
     */
    void predict();

    /**
     * @brief Update camera motion matrix for the track
     * 
     * @param video Video identifier
     * @param frame Frame number
     */
    void cameraUpdate(const std::string& video, int frame);

    /**
     * @brief Perform Kalman filter measurement update step
     * 
     * @param detection Associated detection
     */
    void update(const Detection& detection);

    /**
     * @brief Mark this track as missed (no association at the current time step)
     */
    void markMissed();

    // State query methods
    bool isTentative() const { return state_ == TrackState::Tentative; }
    bool isConfirmed() const { return state_ == TrackState::Confirmed; }
    bool isDeleted() const { return state_ == TrackState::Deleted; }

    // Getters
    int getTrackId() const { return track_id_; }
    int getTimeSinceUpdate() const { return time_since_update_; }
    const Eigen::VectorXf& getMean() const { return mean_; }
    const Eigen::MatrixXf& getCovariance() const { return covariance_; }
    const std::vector<Eigen::VectorXf>& getFeatures() const { return features_; }
    const KalmanFilter& getKalmanFilter() const { return kf_; }

private:
    // Track parameters
    int track_id_;
    int hits_;
    int age_;
    int time_since_update_;
    int n_init_;
    int max_age_;

    // Track state
    TrackState state_;
    std::vector<Eigen::VectorXf> features_;
    std::vector<float> scores_;

    // Kalman filter state
    KalmanFilter kf_;
    Eigen::VectorXf mean_;
    Eigen::MatrixXf covariance_;
};

} // namespace tracking 