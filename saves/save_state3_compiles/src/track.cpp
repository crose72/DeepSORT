#include "track.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace tracking {

Track::Track(const Detection& detection, int track_id, int n_init, int max_age)
    : track_id_(track_id)
    , hits_(1)
    , age_(1)
    , time_since_update_(0)
    , n_init_(n_init)
    , max_age_(max_age)
    , state_(TrackState::Tentative) {
    
    // Initialize feature vector
    Eigen::VectorXf feature = detection.getFeature();
    feature.normalize();
    features_.push_back(feature);

    // Initialize score
    scores_.push_back(detection.getConfidence());

    // Initialize Kalman filter state
    auto [mean, covariance] = kf_.initiate(detection.toXYAH());
    mean_ = mean;
    covariance_ = covariance;
}

Eigen::Vector4f Track::toTLWH() const {
    Eigen::Vector4f ret = mean_.head(4);
    ret(2) *= ret(3);
    ret.head(2) -= ret.tail(2) / 2;
    return ret;
}

Eigen::Vector4f Track::toTLBR() const {
    Eigen::Vector4f ret = toTLWH();
    ret.tail(2) += ret.head(2);
    return ret;
}

void Track::predict() {
    auto [new_mean, new_covariance] = kf_.predict(mean_, covariance_);
    mean_ = new_mean;
    covariance_ = new_covariance;
    age_ += 1;
    time_since_update_ += 1;
}

void Track::cameraUpdate(const std::string& video, int frame) {
    // This is a placeholder for camera motion compensation
    // In the actual implementation, you would need to:
    // 1. Get the camera motion matrix for the given video and frame
    // 2. Apply the transformation to the track's state
    // 3. Update the Kalman filter state accordingly
}

void Track::update(const Detection& detection) {
    auto [new_mean, new_covariance] = kf_.update(mean_, covariance_, 
                                                detection.toXYAH(), 
                                                detection.getConfidence());
    mean_ = new_mean;
    covariance_ = new_covariance;

    // Update feature vector
    Eigen::VectorXf feature = detection.getFeature();
    feature.normalize();

    // Apply EMA if enabled
    static const float EMA_ALPHA = 0.9f;  // This should come from config
    static const bool USE_EMA = true;     // This should come from config

    if (USE_EMA && !features_.empty()) {
        Eigen::VectorXf smooth_feat = EMA_ALPHA * features_.back() + (1 - EMA_ALPHA) * feature;
        smooth_feat.normalize();
        features_ = {smooth_feat};
    } else {
        features_.push_back(feature);
    }

    // Update track state
    hits_ += 1;
    time_since_update_ = 0;
    if (state_ == TrackState::Tentative && hits_ >= n_init_) {
        state_ = TrackState::Confirmed;
    }
}

void Track::markMissed() {
    if (state_ == TrackState::Tentative) {
        state_ = TrackState::Deleted;
    } else if (time_since_update_ > max_age_) {
        state_ = TrackState::Deleted;
    }
}

} // namespace tracking 