#pragma once

#include <Eigen/Dense>
#include <vector>

namespace tracking {

class Detection {
public:
    /**
     * @brief Construct a new Detection object
     * 
     * @param tlwh Bounding box in format (x, y, w, h)
     * @param confidence Detector confidence score
     * @param feature Feature vector describing the object
     */
    Detection(const Eigen::Vector4f& tlwh, float confidence, const Eigen::VectorXf& feature);

    /**
     * @brief Convert bounding box to format (min x, min y, max x, max y)
     * 
     * @return Eigen::Vector4f Bounding box in TLBR format
     */
    Eigen::Vector4f toTLBR() const;

    /**
     * @brief Convert bounding box to format (center x, center y, aspect ratio, height)
     * 
     * @return Eigen::Vector4f Bounding box in XYAH format
     */
    Eigen::Vector4f toXYAH() const;

    // Getters
    const Eigen::Vector4f& getTLWH() const { return tlwh_; }
    float getConfidence() const { return confidence_; }
    const Eigen::VectorXf& getFeature() const { return feature_; }

private:
    Eigen::Vector4f tlwh_;      // Bounding box (top left x, top left y, width, height)
    float confidence_;          // Detector confidence score
    Eigen::VectorXf feature_;   // Feature vector
};

} // namespace tracking 