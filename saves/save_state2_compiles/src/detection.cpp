#include "detection.hpp"

namespace tracking {

Detection::Detection(const Eigen::Vector4f& tlwh, float confidence, const Eigen::VectorXf& feature)
    : tlwh_(tlwh)
    , confidence_(confidence)
    , feature_(feature) {
}

Eigen::Vector4f Detection::toTLBR() const {
    Eigen::Vector4f ret = tlwh_;
    ret.segment<2>(2) += ret.segment<2>(0);  // Add width and height to top-left coordinates
    return ret;
}

Eigen::Vector4f Detection::toXYAH() const {
    Eigen::Vector4f ret = tlwh_;
    ret.segment<2>(0) += ret.segment<2>(2) / 2.0f;  // Convert top-left to center coordinates
    ret(2) /= ret(3);  // width / height for aspect ratio
    return ret;
}

} // namespace tracking 