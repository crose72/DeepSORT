#include "tracker.hpp"
#include <algorithm>

namespace tracking {

Tracker::Tracker(matching::NearestNeighborDistanceMetric metric,
                float max_iou_distance,
                int max_age,
                int n_init)
    : next_id_(1)
    , metric_(std::move(metric))
    , max_iou_distance_(max_iou_distance)
    , max_age_(max_age)
    , n_init_(n_init) {
}

void Tracker::predict() {
    for (auto& track : tracks_) {
        track.predict();
    }
}

void Tracker::cameraUpdate(const std::string& video, int frame) {
    for (auto& track : tracks_) {
        track.cameraUpdate(video, frame);
    }
}

void Tracker::update(const std::vector<Detection>& detections) {
    // Run matching cascade
    auto [matches, unmatched_tracks, unmatched_detections] = match(detections);

    // Update track set
    for (const auto& [track_idx, detection_idx] : matches) {
        tracks_[track_idx].update(detections[detection_idx]);
    }

    for (int track_idx : unmatched_tracks) {
        tracks_[track_idx].markMissed();
    }

    for (int detection_idx : unmatched_detections) {
        initiateTrack(detections[detection_idx]);
    }

    // Remove deleted tracks
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                      [](const Track& t) { return t.isDeleted(); }),
        tracks_.end());

    // Update distance metric
    std::vector<int> active_targets;
    std::vector<Eigen::VectorXf> features;
    std::vector<int> targets;

    for (const auto& track : tracks_) {
        if (!track.isConfirmed()) continue;

        active_targets.push_back(track.getTrackId());
        features.insert(features.end(),
                       track.getFeatures().begin(),
                       track.getFeatures().end());
        targets.insert(targets.end(),
                      track.getFeatures().size(),
                      track.getTrackId());
    }

    metric_.partialFit(features, targets, active_targets);
}

void Tracker::initiateTrack(const Detection& detection) {
    tracks_.emplace_back(detection, next_id_, n_init_, max_age_);
    next_id_ += 1;
}

std::tuple<std::vector<std::pair<int, int>>, std::vector<int>, std::vector<int>>
Tracker::match(const std::vector<Detection>& detections) {
    std::vector<int> confirmed_tracks;
    std::vector<int> unconfirmed_tracks;

    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (tracks_[i].isConfirmed()) {
            confirmed_tracks.push_back(i);
        } else {
            unconfirmed_tracks.push_back(i);
        }
    }

    // Associate confirmed tracks using appearance features
    auto gated_metric = [this](const std::vector<Track>& tracks,
                              const std::vector<Detection>& dets,
                              const std::vector<int>& track_indices,
                              const std::vector<int>& detection_indices) {
        std::vector<Eigen::VectorXf> features;
        std::vector<int> targets;

        for (int i : detection_indices) {
            features.push_back(dets[i].getFeature());
        }
        for (int i : track_indices) {
            targets.push_back(tracks[i].getTrackId());
        }

        Eigen::MatrixXf cost_matrix = metric_.distance(features, targets);
        return matching::gateCostMatrix(cost_matrix, tracks, dets,
                                      track_indices, detection_indices);
    };

    auto [matches_a, unmatched_tracks_a, unmatched_detections] =
        matching::matchingCascade(gated_metric, metric_.getMatchingThreshold(),
                                max_age_, tracks_, detections, confirmed_tracks);

    // Associate remaining tracks together with unconfirmed tracks using IOU
    std::vector<int> iou_track_candidates = unconfirmed_tracks;
    for (int idx : unmatched_tracks_a) {
        if (tracks_[idx].getTimeSinceUpdate() == 1) {
            iou_track_candidates.push_back(idx);
        }
    }

    std::vector<int> unmatched_tracks_a_1;
    for (int idx : unmatched_tracks_a) {
        if (tracks_[idx].getTimeSinceUpdate() != 1) {
            unmatched_tracks_a_1.push_back(idx);
        }
    }

    auto [matches_b, unmatched_tracks_b, unmatched_detections_b] =
        matching::minCostMatching(matching::iouCost, max_iou_distance_,
                                tracks_, detections,
                                iou_track_candidates,
                                unmatched_detections);

    std::vector<std::pair<int, int>> matches;
    matches.insert(matches.end(), matches_a.begin(), matches_a.end());
    matches.insert(matches.end(), matches_b.begin(), matches_b.end());

    std::vector<int> unmatched_tracks;
    unmatched_tracks.insert(unmatched_tracks.end(),
                           unmatched_tracks_a_1.begin(),
                           unmatched_tracks_a_1.end());
    unmatched_tracks.insert(unmatched_tracks.end(),
                           unmatched_tracks_b.begin(),
                           unmatched_tracks_b.end());

    return std::make_tuple(matches, unmatched_tracks, unmatched_detections_b);
}

} // namespace tracking 