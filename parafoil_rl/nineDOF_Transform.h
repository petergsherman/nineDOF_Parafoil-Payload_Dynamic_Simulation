#ifndef NINEDOF_TRANSFORM_H
#define NINEDOF_TRANSFORM_H

#include <array>
#include <cmath>

using Matrix3x3 = std::array<std::array<double, 3>, 3>;
using Vector3 = std::array<double, 3>;

// Create transformation matrix from inertial to body frame
inline Matrix3x3 makeTransform_InertialToBody(double phi, double theta, double psi) {
    double cphi = std::cos(phi);
    double sphi = std::sin(phi);
    double ctheta = std::cos(theta);
    double stheta = std::sin(theta);
    double cpsi = std::cos(psi);
    double spsi = std::sin(psi);
    
    Matrix3x3 T = {{
        {ctheta*cpsi, sphi*stheta*cpsi - cphi*spsi, cphi*stheta*cpsi + sphi*spsi},
        {ctheta*spsi, sphi*stheta*spsi + cphi*cpsi, cphi*stheta*spsi - sphi*cpsi},
        {-stheta,     sphi*ctheta,                  cphi*ctheta}
    }};
    
    return T;
}

// Euler-rate Jacobian Tensor
inline Matrix3x3 makeJ(double phi, double theta, double psi) {
    double c_phi = std::cos(phi);
    double c_theta = std::cos(theta);
    double s_phi = std::sin(phi);
    double t_theta = std::tan(theta);
    
    Matrix3x3 J = {{
        {1.0, s_phi * t_theta, c_phi * t_theta},
        {0.0, c_phi,           -s_phi},
        {0.0, s_phi / c_theta, c_phi / c_theta}
    }};
    
    return J;
}

#endif // NINEDOF_TRANSFORM_H