#pragma once

#include "sophus/so3.hpp"

namespace Sophus {

template <typename SO3Type>
typename SO3Type::Transformation jl(const typename SO3Type::Point& Omega) {
    using Scalar = typename SO3Type::Scalar;
    using Transformation = typename SO3Type::Transformation;
    using Point = typename SO3Type::Point;

    Scalar theta = Omega.norm();
    if (theta < 1e-6) {
        return Transformation::Identity();
    }

    Point a = Omega;
    a.normalize();
    double sin_theta = std::sin(theta);
    double cos_theta = std::cos(theta);
    return (sin_theta / theta) * Transformation ::Identity() + (1 - sin_theta / theta) * a * a.transpose() +
           (1 - cos_theta) / theta * SO3Type::hat(a);
}

template <typename SO3Type>
typename SO3Type::Transformation jl_inv(const typename SO3Type::Point& Omega) {
    using Scalar = typename SO3Type::Scalar;
    using Transformation = typename SO3Type::Transformation;
    using Point = typename SO3Type::Point;

    Scalar theta = Omega.norm();
    if (theta < 1e-6) {
        return Transformation::Identity();
    }

    Point a = Omega;
    a.normalize();

    double cot_half_theta = cos(0.5 * theta) / sin(0.5 * theta);
    return 0.5 * theta * cot_half_theta * Transformation::Identity() +
           (1 - 0.5 * theta * cot_half_theta) * a * a.transpose() - 0.5 * theta * SO3Type::hat(a);
}

template <typename SO3Type>
typename SO3Type::Transformation jr(const typename SO3Type::Point& Omega) {
    return jl<SO3Type>(-Omega);
}
template <typename SO3Type>
typename SO3Type::Transformation jr(const SO3Type& R) {
    return jr<SO3Type>(R.log());
}

template <typename SO3Type>
typename SO3Type::Transformation jr_inv(const typename SO3Type::Point& Omega) {
    return jl_inv<SO3Type>(-Omega);
}

template <typename SO3Type>
typename SO3Type::Transformation jr_inv(const SO3Type& R) {
    return jr_inv<SO3Type>(R.log());
}
}  // namespace Sophus
