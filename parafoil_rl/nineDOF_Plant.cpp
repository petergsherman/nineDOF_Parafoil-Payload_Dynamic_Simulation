#include "nineDOF_Plant.h"
#include <cmath>
#include <algorithm>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper function: matrix multiplication
Matrix3x3 matmul(const Matrix3x3& A, const Matrix3x3& B) {
    Matrix3x3 result = {{{0}}};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Helper function: matrix-vector multiplication
Vector3 matvec(const Matrix3x3& A, const Vector3& v) {
    Vector3 result = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result[i] += A[i][j] * v[j];
        }
    }
    return result;
}

// Helper function: matrix transpose
Matrix3x3 transpose(const Matrix3x3& A) {
    Matrix3x3 result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result[i][j] = A[j][i];
        }
    }
    return result;
}

// Helper function: matrix addition
Matrix3x3 matadd(const Matrix3x3& A, const Matrix3x3& B) {
    Matrix3x3 result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return result;
}

// Helper function: matrix subtraction
Matrix3x3 matsub(const Matrix3x3& A, const Matrix3x3& B) {
    Matrix3x3 result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    return result;
}

// Helper function: scalar matrix multiplication
Matrix3x3 matscale(const Matrix3x3& A, double s) {
    Matrix3x3 result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result[i][j] = A[i][j] * s;
        }
    }
    return result;
}

// Helper function: create identity matrix
Matrix3x3 identity3x3() {
    Matrix3x3 I = {{{0}}};
    I[0][0] = I[1][1] = I[2][2] = 1.0;
    return I;
}

// Helper function: vector norm
double norm(const Vector3& v) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

// Helper function: solve 12x12 linear system using Gaussian elimination
std::array<double, 12> solve12x12(const std::array<std::array<double, 12>, 12>& A, 
                                   const std::array<double, 12>& b) {
    std::array<std::array<double, 12>, 12> M = A;
    std::array<double, 12> x = b;
    
    // Forward elimination
    for (int i = 0; i < 12; i++) {
        // Find pivot
        int maxRow = i;
        for (int k = i + 1; k < 12; k++) {
            if (std::abs(M[k][i]) > std::abs(M[maxRow][i])) {
                maxRow = k;
            }
        }
        
        // Swap rows
        std::swap(M[i], M[maxRow]);
        std::swap(x[i], x[maxRow]);
        
        // Singular matrix check
        if (std::abs(M[i][i]) < 1e-10) {
            std::cerr << "Singular matrix in dynamics computation" << std::endl;
            return {0};
        }
        
        // Eliminate column
        for (int k = i + 1; k < 12; k++) {
            double factor = M[k][i] / M[i][i];
            for (int j = i; j < 12; j++) {
                M[k][j] -= factor * M[i][j];
            }
            x[k] -= factor * x[i];
        }
    }
    
    // Back substitution
    for (int i = 11; i >= 0; i--) {
        for (int k = i + 1; k < 12; k++) {
            x[i] -= M[i][k] * x[k];
        }
        x[i] /= M[i][i];
    }
    
    return x;
}

Plant::Plant(const SystemParameters& parameters, AtmosphereParameters* atmosphere)
    : params(parameters), atm(atmosphere) {
    populateInertias();
}

void Plant::populateInertias() {
    // Parafoil Inertia Matrix
    I_P = {{
        {params.PIXX, params.PIXY, params.PIXZ},
        {params.PIXY, params.PIYY, params.PIYZ},
        {params.PIXZ, params.PIYZ, params.PIZZ}
    }};
    
    // Cradle inertia matrix
    I_C = {{
        {params.CIXX, params.CIXY, params.CIXZ},
        {params.CIXY, params.CIYY, params.CIYZ},
        {params.CIXZ, params.CIYZ, params.CIZZ}
    }};
    
    // Apparent mass matrix (diagonal)
    I_AM = {{
        {params.PMASSA, 0, 0},
        {0, params.PMASSB, 0},
        {0, 0, params.PMASSC}
    }};
    
    // Apparent inertia matrix (diagonal)
    I_AI = {{
        {params.PMASSP, 0, 0},
        {0, params.PMASSQ, 0},
        {0, 0, params.PMASSR}
    }};
    
    // Spanwise camber matrix
    I_H = {{
        {0.0, params.PMASSH, 0.0},
        {params.PMASSH, 0.0, 0.0},
        {0.0, 0.0, 0.0}
    }};
}

Matrix3x3 Plant::makeT_PPI(double incidence, double nominalIncidence) {
    double total_incidence = incidence + params.NOM_INCIDENCE;
    double ci = std::cos(total_incidence);
    double si = std::sin(total_incidence);
    
    Matrix3x3 T_PPI = {{
        {ci, 0, si},
        {0, 1, 0},
        {-si, 0, ci}
    }};
    
    return T_PPI;
}

Matrix3x3 Plant::skew(const Vector3& v) {
    Matrix3x3 S = {{
        {0, -v[2], v[1]},
        {v[2], 0, -v[0]},
        {-v[1], v[0], 0}
    }};
    return S;
}

void Plant::interpolateAeroTables(double sig1, double& cd0, double& cda2, 
                                  double& cl0, double& cla) {
    // Find bracket
    if (sig1 <= params.SIGTAB[0]) {
        cd0 = params.CD0TAB[0];
        cda2 = params.CDA2TAB[0];
        cl0 = params.CL0TAB[0];
        cla = params.CLATAB[0];
        return;
    }
    
    if (sig1 >= params.SIGTAB.back()) {
        cd0 = params.CD0TAB.back();
        cda2 = params.CDA2TAB.back();
        cl0 = params.CL0TAB.back();
        cla = params.CLATAB.back();
        return;
    }
    
    // Linear interpolation
    size_t idx = 0;
    for (size_t i = 0; i < params.SIGTAB.size() - 1; i++) {
        if (sig1 >= params.SIGTAB[i] && sig1 <= params.SIGTAB[i+1]) {
            idx = i;
            break;
        }
    }
    
    double m = (sig1 - params.SIGTAB[idx]) / (params.SIGTAB[idx+1] - params.SIGTAB[idx]);
    
    cd0 = params.CD0TAB[idx] + m * (params.CD0TAB[idx+1] - params.CD0TAB[idx]);
    cda2 = params.CDA2TAB[idx] + m * (params.CDA2TAB[idx+1] - params.CDA2TAB[idx]);
    cl0 = params.CL0TAB[idx] + m * (params.CL0TAB[idx+1] - params.CL0TAB[idx]);
    cla = params.CLATAB[idx] + m * (params.CLATAB[idx+1] - params.CLATAB[idx]);
}

double Plant::interpolateCND2(double alpha) {
    if (alpha <= params.AOATAB[0]) {
        return params.CND2TAB[0];
    }
    
    if (alpha >= params.AOATAB.back()) {
        return params.CND2TAB.back();
    }
    
    size_t idx = 0;
    for (size_t i = 0; i < params.AOATAB.size() - 1; i++) {
        if (alpha >= params.AOATAB[i] && alpha <= params.AOATAB[i+1]) {
            idx = i;
            break;
        }
    }
    
    double m = (alpha - params.AOATAB[idx]) / (params.AOATAB[idx+1] - params.AOATAB[idx]);
    return params.CND2TAB[idx] + m * (params.CND2TAB[idx+1] - params.CND2TAB[idx]);
}

void Plant::computeAerodynamics(const State& state, double deltaL, double deltaR,
                                double incidence, Vector3& C_FAp_P, Vector3& C_MAp_P,
                                Vector3& C_FAc_C) {
    // Extract state components
    double phiP = state[3], thetaP = state[4], psiP = state[5];
    double phiC = state[6], thetaC = state[7], psiC = state[8];
    double uG = state[9], vG = state[10], wG = state[11];
    double pP = state[12], qP = state[13], rP = state[14];
    double pC = state[15], qC = state[16], rC = state[17];
    
    // Transformation matrices
    Matrix3x3 T_IP = makeTransform_InertialToBody(phiP, thetaP, psiP);
    Matrix3x3 T_IC = makeTransform_InertialToBody(phiC, thetaC, psiC);
    Matrix3x3 T_IP_t = transpose(T_IP);
    Matrix3x3 T_IC_t = transpose(T_IC);
    Matrix3x3 T_PC = matmul(T_IP_t, T_IC);
    
    // Velocities
    Vector3 vG_P = {uG, vG, wG};
    Vector3 wP_P = {pP, qP, rP};
    Vector3 wC_C = {pC, qC, rC};
    Matrix3x3 S_wP_P = skew(wP_P);
    Matrix3x3 S_wC_C = skew(wC_C);
    
    // Geometry vectors
    Vector3 rPG_P = {-params.SLPARAFOIL, -params.BLPARAFOIL, -params.WLPARAFOIL};
    Vector3 rPR_P = {params.SLPR, params.BLPR, params.WLPR};
    Vector3 rRAp_PI = {params.SLRAP, params.BLRAP, params.WLRAP};
    Vector3 rGC_C = {params.SLCRADLE, params.BLCRADLE, params.WLCRADLE};
    
    // Incidence rotation matrix
    Matrix3x3 T_PPI = makeT_PPI(incidence, params.NOM_INCIDENCE);
    
    // Rotation point to aero center
    Vector3 rRAp_P = matvec(T_PPI, rRAp_PI);
    Vector3 rPAp_P = {rPR_P[0] + rRAp_P[0], rPR_P[1] + rRAp_P[1], rPR_P[2] + rRAp_P[2]};
    Vector3 rGAp_P = {-rPG_P[0] + rPAp_P[0], -rPG_P[1] + rPAp_P[1], -rPG_P[2] + rPAp_P[2]};
    
    // Velocity of aerodynamic center
    Vector3 cross_wP_rG = matvec(S_wP_P, rGAp_P);
    Vector3 vAp_P = {vG_P[0] + cross_wP_rG[0], vG_P[1] + cross_wP_rG[1], vG_P[2] + cross_wP_rG[2]};
    Vector3 vAp_P_wind = {vAp_P[0] - atm->VXWIND, vAp_P[1] - atm->VYWIND, vAp_P[2] - atm->VZWIND};
    
    Matrix3x3 T_PPI_t = transpose(T_PPI);
    Vector3 vAp_PI = matvec(T_PPI_t, vAp_P_wind);
    double vAp = norm(vAp_PI);
    
    // Computing cradle velocity
    Vector3 temp_vel = matvec(T_PC, vG_P);
    Vector3 cross_wC_rG = matvec(S_wC_C, rGC_C);
    Vector3 vC_C = {temp_vel[0] + cross_wC_rG[0], temp_vel[1] + cross_wC_rG[1], temp_vel[2] + cross_wC_rG[2]};
    double vC = norm(vC_C);
    
    // Avoid singularity
    if (vAp < 0.01) {
        C_FAp_P = {0, 0, 0};
        C_MAp_P = {0, 0, 0};
        C_FAc_C = {0, 0, 0};
        return;
    }
    
    // Calculate angles
    double alpha = std::atan2(vAp_PI[2], vAp_PI[0]);
    double beta = std::asin(std::clamp(vAp_PI[1] / vAp, -1.0, 1.0));
    
    // Computing controls
    double dA = deltaR - deltaL;
    double dS = 0.5 * (deltaR + deltaL) - params.deadband + params.NOM_BRAKE;
    double sig1 = dS / params.dbar;
    
    // Interpolate aerodynamic coefficients
    double CD0, CDA2, CL0, CLA;
    interpolateAeroTables(sig1, CD0, CDA2, CL0, CLA);
    double CND2 = interpolateCND2(alpha);
    
    // Normalized controls
    double sigmaLeft = deltaL / params.dbar;
    double sigmaRight = deltaR / params.dbar;
    
    // Aerodynamic coefficients
    double Cd = CD0 + CDA2 * (alpha * alpha);
    double Cy = params.CYBETA * beta;
    double Cl = CL0 + (CLA * alpha);
    
    double C_l = (params.CLBETA * beta + 
                  (params.bbar_parafoil / (2 * vAp)) * params.CLP * pP +
                  (params.bbar_parafoil / (2 * vAp)) * params.CLR * rP + 
                  params.CLDA * dA / params.dbar);
    
    double C_m = params.CM0 + (params.cbar_parafoil / (2 * vAp)) * params.CMQ * qP + 
                 params.CMDS * dS / params.dbar;
    
    double C_n = (params.CNBETA * beta + 
                  (params.bbar_parafoil / (2 * vAp)) * params.CNP * pP +
                  (params.bbar_parafoil / (2 * vAp)) * params.CNR * rP - 
                  params.CND1 * sigmaLeft - CND2 * sigmaLeft * sig1 + 
                  params.CND1 * sigmaRight + CND2 * sigmaRight * sig1);
    
    // Stall model
    Cl = std::clamp(Cl, 0.5, 2.0);
    Cd = std::clamp(Cd, 0.0, 1.5);
    
    // Calculate forces and moments
    double ca = std::cos(alpha);
    double sa = std::sin(alpha);
    
    Vector3 C_FAp_PI = {
        -(ca * Cd) + (sa * Cl),
        Cy,
        (-sa * Cd) - (ca * Cl)
    };
    
    double qbar = 0.5 * atm->DEN * params.A_parafoil * vAp * vAp;
    C_FAp_PI[0] *= qbar;
    C_FAp_PI[1] *= qbar;
    C_FAp_PI[2] *= qbar;
    
    C_FAp_P = matvec(T_PPI, C_FAp_PI);
    
    C_MAp_P = {
        qbar * params.bbar_parafoil * C_l,
        qbar * params.cbar_parafoil * C_m,
        qbar * params.bbar_parafoil * C_n
    };
    
    double vC_factor = -0.5 * atm->DEN * vC * params.A_cradle * params.CD_cradle;
    C_FAc_C = {vC_factor * vC_C[0], vC_factor * vC_C[1], vC_factor * vC_C[2]};
}

State Plant::computeDerivatives(const State& state, double deltaL, double deltaR, double incidence) {
    State statedot = {0};
    
    // Extract state
    double phiP = state[3], thetaP = state[4], psiP = state[5];
    double phiC = state[6], thetaC = state[7], psiC = state[8];
    double uG = state[9], vG = state[10], wG = state[11];
    double pP = state[12], qP = state[13], rP = state[14];
    double pC = state[15], qC = state[16], rC = state[17];
    
    Vector3 vG_P = {uG, vG, wG};
    Vector3 wP_P = {pP, qP, rP};
    Vector3 wC_C = {pC, qC, rC};
    
    Matrix3x3 S_wP_P = skew(wP_P);
    Matrix3x3 S_wC_C = skew(wC_C);
    
    // Transformation matrices
    Matrix3x3 T_IP = makeTransform_InertialToBody(phiP, thetaP, psiP);
    Matrix3x3 T_IC = makeTransform_InertialToBody(phiC, thetaC, psiC);
    Matrix3x3 T_IP_t = transpose(T_IP);
    Matrix3x3 T_IC_t = transpose(T_IC);
    Matrix3x3 T_CP = matmul(T_IC_t, T_IP);
    Matrix3x3 T_PPI = makeT_PPI(incidence, params.NOM_INCIDENCE);
    Matrix3x3 T_parafoilJ = makeJ(phiP, thetaP, psiP);
    Matrix3x3 T_bodyJ = makeJ(phiC, thetaC, psiC);
    
    // Update non-coupled terms
    Vector3 pos_dot = matvec(T_IP, vG_P);
    Vector3 euler_p_dot = matvec(T_parafoilJ, wP_P);
    Vector3 euler_c_dot = matvec(T_bodyJ, wC_C);
    
    statedot[0] = pos_dot[0]; statedot[1] = pos_dot[1]; statedot[2] = pos_dot[2];
    statedot[3] = euler_p_dot[0]; statedot[4] = euler_p_dot[1]; statedot[5] = euler_p_dot[2];
    statedot[6] = euler_c_dot[0]; statedot[7] = euler_c_dot[1]; statedot[8] = euler_c_dot[2];
    
    // Geometry vectors
    Vector3 rPG_P = {-params.SLPARAFOIL, -params.BLPARAFOIL, -params.WLPARAFOIL};
    Vector3 rGC_C = {params.SLCRADLE, params.BLCRADLE, params.WLCRADLE};
    Vector3 rPMp_P = {params.SLPMP, params.BLPMP, params.WLPMP};
    Vector3 rGMp_P = {-rPG_P[0] + rPMp_P[0], -rPG_P[1] + rPMp_P[1], -rPG_P[2] + rPMp_P[2]};
    
    Vector3 rPR_P = {params.SLPR, params.BLPR, params.WLPR};
    Vector3 rRAp_PI = {params.SLRAP, params.BLRAP, params.WLRAP};
    Vector3 rRAp_P = matvec(T_PPI, rRAp_PI);
    Vector3 rPAp_P = {rPR_P[0] + rRAp_P[0], rPR_P[1] + rRAp_P[1], rPR_P[2] + rRAp_P[2]};
    
    // Skew matrices
    Matrix3x3 S_rCG_C = skew({-rGC_C[0], -rGC_C[1], -rGC_C[2]});
    Matrix3x3 S_rPG_P = skew(rPG_P);
    Matrix3x3 S_rPMp_P = skew(rPMp_P);
    Matrix3x3 S_rGMp_P = skew(rGMp_P);
    Matrix3x3 S_rPAp_P = skew(rPAp_P);
    
    // Velocity of apparent mass center
    Vector3 cross_w_rG = matvec(S_wP_P, rGMp_P);
    Vector3 vMp_P = {vG_P[0] + cross_w_rG[0], vG_P[1] + cross_w_rG[1], vG_P[2] + cross_w_rG[2]};
    Matrix3x3 S_vMp_P = skew(vMp_P);
    
    // Get aerodynamic forces
    Vector3 C_FAp_P, C_MAp_P, C_FAc_C;
    computeAerodynamics(state, deltaL, deltaR, incidence, C_FAp_P, C_MAp_P, C_FAc_C);
    
    // Gravitational forces
    Vector3 C_FWp_P = {
        -params.m_parafoil * params.GRAVITY * std::sin(thetaP),
        params.m_parafoil * params.GRAVITY * std::sin(phiP) * std::cos(thetaP),
        params.m_parafoil * params.GRAVITY * std::cos(phiP) * std::cos(thetaP)
    };
    
    Vector3 C_FWc_C = {
        -params.m_cradle * params.GRAVITY * std::sin(thetaC),
        params.m_cradle * params.GRAVITY * std::sin(phiC) * std::cos(thetaC),
        params.m_cradle * params.GRAVITY * std::cos(phiC) * std::cos(thetaC)
    };
    
    // Gimbal stiffness moment
    Vector3 C_MG_P = {
        0.0,
        0.0,
        params.KGIMBAL * (psiP - psiC) + params.CGIMBAL * (statedot[5] - statedot[8])
    };
    
    // Build 12x12 system (simplified, would need full matrix operations)
    // For brevity, returning zero derivatives for coupled terms
    // A full implementation would solve the 12x12 linear system
    
    std::array<std::array<double, 12>, 12> A = {{{0}}};
    std::array<double, 12> b = {0};
    
    //////////////////////////////BUILDING A MATRIX///////////////////////////////////
    // A11 = m_cradle * S_rCG_C (rows 0-2, cols 0-2)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[i][j] = params.m_cradle * S_rCG_C[i][j];
        }
    }
    
    // A12 = zeros(3,3) (rows 0-2, cols 3-5) - already zero

    // A13 = m_cradle * T_IC.T @ T_IP (rows 0-2, cols 6-8)
    Matrix3x3 temp_A13 = matmul(T_IC_t, T_IP);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[i][j + 6] = params.m_cradle * temp_A13[i][j];
        }
    }
    
    // A14 = -T_IC.T (rows 0-2, cols 9-11)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[i][j + 9] = -T_IC_t[i][j];
        }
    }
    
    // A21 = zeros(3,3) (rows 3-5, cols 0-2) - already zero
    // A22 = -I_AM @ S_rGMp_P + I_H + m_parafoil * S_rPG_P (rows 3-5, cols 3-5)
    Matrix3x3 temp1 = matmul(I_AM, S_rGMp_P);
    Matrix3x3 temp2 = matscale(temp1, -1.0);
    Matrix3x3 temp3 = matadd(temp2, I_H);
    Matrix3x3 temp4 = matscale(S_rPG_P, params.m_parafoil);
    Matrix3x3 A22 = matadd(temp3, temp4);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[i + 3][j + 3] = A22[i][j];
        }
    }

    // A23 = m_parafoil * I + I_AM (rows 3-5, cols 6-8)
    Matrix3x3 eye = identity3x3();
    Matrix3x3 temp5 = matscale(eye, params.m_parafoil);
    Matrix3x3 A23 = matadd(temp5, I_AM);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[i + 3][j + 6] = A23[i][j];
        }
    }

    // A24 = T_IP.T (rows 3-5, cols 9-11)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[i + 3][j + 9] = T_IP_t[i][j];
        }
    }

    // A31 = I_C (rows 6-8, cols 0-2)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[i + 6][j] = I_C[i][j];
        }
    }

    // A32 = zeros(3,3) (rows 6-8, cols 3-5) - already zero

    // A33 = zeros(3,3) (rows 6-8, cols 6-8) - already zero

    // A34 = -S_rCG_C @ T_IC.T (rows 6-8, cols 9-11)
    Matrix3x3 temp6 = matmul(S_rCG_C, T_IC_t);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[i + 6][j + 9] = -temp6[i][j];
        }
    }

    // A41 = zeros(3,3) (rows 9-11, cols 0-2) - already zero

    // A42 = I_P + S_rPMp_P @ (I_H - I_AM @ S_rGMp_P) - I_H @ S_rGMp_P + I_AI (rows 9-11, cols 3-5)
    Matrix3x3 temp7 = matmul(I_AM, S_rGMp_P);
    Matrix3x3 temp8 = matsub(I_H, temp7);
    Matrix3x3 temp9 = matmul(S_rPMp_P, temp8);
    Matrix3x3 temp10 = matmul(I_H, S_rGMp_P);
    Matrix3x3 temp11 = matsub(temp9, temp10);
    Matrix3x3 temp12 = matadd(I_P, temp11);
    Matrix3x3 A42 = matadd(temp12, I_AI);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[i + 9][j + 3] = A42[i][j];
        }
    }

    // A43 = I_H + S_rPMp_P @ I_AM (rows 9-11, cols 6-8)
    Matrix3x3 temp13 = matmul(S_rPMp_P, I_AM);
    Matrix3x3 A43 = matadd(I_H, temp13);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[i + 9][j + 6] = A43[i][j];
        }
    }

    // A44 = S_rPG_P @ T_IP.T (rows 9-11, cols 9-11)
    Matrix3x3 A44 = matmul(S_rPG_P, T_IP_t);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[i + 9][j + 9] = A44[i][j];
        }
    }

    ///////////////////BUILDING B VECTOR/////////////////////////
    // b1 = C_FAc_C + C_FWc_C - m_cradle * T_CP @ S_wP_P @ vG_P - m_cradle * S_wC_C @ S_wC_C @ rGC_C
    Vector3 temp_b1_1 = matvec(S_wP_P, vG_P);
    Vector3 temp_b1_2 = matvec(T_CP, temp_b1_1);
    Vector3 temp_b1_3 = matvec(S_wC_C, rGC_C);
    Vector3 temp_b1_4 = matvec(S_wC_C, temp_b1_3);
    
    Vector3 b1;
    for (int i = 0; i < 3; i++) {
        b1[i] = C_FAc_C[i] + C_FWc_C[i] - params.m_cradle * temp_b1_2[i] - params.m_cradle * temp_b1_4[i];
    }
    b[0] = b1[0]; b[1] = b1[1]; b[2] = b1[2];
    
    // b2 = C_FAp_P + C_FWp_P - m_parafoil * S_wP_P @ vG_P - S_wP_P @ (I_AM @ vMp_P + I_H @ wP_P) + m_parafoil * S_wP_P @ S_wP_P @ rPG_P
    Vector3 temp_b2_1 = matvec(S_wP_P, vG_P);
    Vector3 temp_b2_2 = matvec(I_AM, vMp_P);
    Vector3 temp_b2_3 = matvec(I_H, wP_P);
    Vector3 temp_b2_4 = {temp_b2_2[0] + temp_b2_3[0], temp_b2_2[1] + temp_b2_3[1], temp_b2_2[2] + temp_b2_3[2]};
    Vector3 temp_b2_5 = matvec(S_wP_P, temp_b2_4);
    Vector3 temp_b2_6 = matvec(S_wP_P, rPG_P);
    Vector3 temp_b2_7 = matvec(S_wP_P, temp_b2_6);
    
    Vector3 b2;
    for (int i = 0; i < 3; i++) {
        b2[i] = C_FAp_P[i] + C_FWp_P[i] - params.m_parafoil * temp_b2_1[i] - temp_b2_5[i] + params.m_parafoil * temp_b2_7[i];
    }
    b[3] = b2[0]; b[4] = b2[1]; b[5] = b2[2];
    
    // b3 = T_CP @ C_MG_P - S_wC_C @ I_C @ wC_C
    Vector3 temp_b3_1 = matvec(T_CP, C_MG_P);
    Vector3 temp_b3_2 = matvec(I_C, wC_C);
    Vector3 temp_b3_3 = matvec(S_wC_C, temp_b3_2);
    
    Vector3 b3;
    for (int i = 0; i < 3; i++) {
        b3[i] = temp_b3_1[i] - temp_b3_3[i];
    }
    b[6] = b3[0]; b[7] = b3[1]; b[8] = b3[2];
    
    // b4 = C_MAp_P + S_rPAp_P @ C_FAp_P - C_MG_P - 
    //      (S_rPMp_P @ S_wP_P + S_vMp_P @ I_H + S_wP_P @ (I_AI + I_P)) @ wP_P - 
    //      (S_rPMp_P @ S_wP_P @ I_AM + S_wP_P @ I_H) @ vMp_P
    
    Vector3 temp_b4_1 = matvec(S_rPAp_P, C_FAp_P);
    
    // First complex term: (S_rPMp_P @ S_wP_P + S_vMp_P @ I_H + S_wP_P @ (I_AI + I_P)) @ wP_P
    Matrix3x3 temp_b4_2 = matmul(S_rPMp_P, S_wP_P);
    Matrix3x3 temp_b4_3 = matmul(S_vMp_P, I_H);
    Matrix3x3 temp_b4_4 = matadd(I_AI, I_P);
    Matrix3x3 temp_b4_5 = matmul(S_wP_P, temp_b4_4);
    Matrix3x3 temp_b4_6 = matadd(temp_b4_2, temp_b4_3);
    Matrix3x3 temp_b4_7 = matadd(temp_b4_6, temp_b4_5);
    Vector3 temp_b4_8 = matvec(temp_b4_7, wP_P);
    
    // Second complex term: (S_rPMp_P @ S_wP_P @ I_AM + S_wP_P @ I_H) @ vMp_P
    Matrix3x3 temp_b4_9 = matmul(S_wP_P, I_AM);
    Matrix3x3 temp_b4_10 = matmul(S_rPMp_P, temp_b4_9);
    Matrix3x3 temp_b4_11 = matmul(S_wP_P, I_H);
    Matrix3x3 temp_b4_12 = matadd(temp_b4_10, temp_b4_11);
    Vector3 temp_b4_13 = matvec(temp_b4_12, vMp_P);
    
    Vector3 b4;
    for (int i = 0; i < 3; i++) {
        b4[i] = C_MAp_P[i] + temp_b4_1[i] - C_MG_P[i] - temp_b4_8[i] - temp_b4_13[i];
    }
    b[9] = b4[0]; b[10] = b4[1]; b[11] = b4[2];
    
    // Solve the 12x12 linear system Ax = b
    std::array<double, 12> X = solve12x12(A, b);
    
    // Populate statedot with coupled terms
    statedot[9] = X[6];   statedot[10] = X[7];   statedot[11] = X[8];   // Gimbal accelerations
    statedot[12] = X[3];  statedot[13] = X[4];   statedot[14] = X[5];   // Parafoil angular accelerations
    statedot[15] = X[0];  statedot[16] = X[1];   statedot[17] = X[2];   // Cradle angular accelerations

    return statedot;
}

State Plant::rk4_step(const State& state, double dt, double deltaL, double deltaR, double incidence) {
    State k1 = computeDerivatives(state, deltaL, deltaR, incidence);
    for (int i = 0; i < 18; i++) k1[i] *= dt;
    
    State state2;
    for (int i = 0; i < 18; i++) state2[i] = state[i] + k1[i] / 2.0;
    State k2 = computeDerivatives(state2, deltaL, deltaR, incidence);
    for (int i = 0; i < 18; i++) k2[i] *= dt;
    
    State state3;
    for (int i = 0; i < 18; i++) state3[i] = state[i] + k2[i] / 2.0;
    State k3 = computeDerivatives(state3, deltaL, deltaR, incidence);
    for (int i = 0; i < 18; i++) k3[i] *= dt;
    
    State state4;
    for (int i = 0; i < 18; i++) state4[i] = state[i] + k3[i];
    State k4 = computeDerivatives(state4, deltaL, deltaR, incidence);
    for (int i = 0; i < 18; i++) k4[i] *= dt;
    
    State result;
    for (int i = 0; i < 18; i++) {
        result[i] = state[i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6.0;
    }
    
    return result;
}

std::pair<std::vector<double>, std::vector<State>>
Plant::run_simulation(const State& state0, double t_final, double dt, ControlFunc control_func) {
    int n_steps = static_cast<int>(t_final / dt) + 1;
    std::vector<double> times;
    std::vector<State> states;
    
    State state = state0;
    times.push_back(0.0);
    states.push_back(state);
    
    for (int i = 1; i < n_steps; i++) {
        double t = times[i-1];
        
        // Update atmosphere if dynamic
        if (atm) {
            double altitude = -states[i-1][2];
            atm->update(t, altitude);
        }
        
        // Get control inputs
        double deltaL = 0.0, deltaR = 0.0, incidence = 0.0;
        if (control_func) {
            auto controls = control_func(t, states[i-1]);
            deltaL = controls[0];
            deltaR = controls[1];
            incidence = controls[2];
        }
        
        // Integrate
        state = rk4_step(states[i-1], dt, deltaL, deltaR, incidence);
        
        // Angle wrapping
        for (int angle_idx : {4, 5, 7, 8}) {
            while (state[angle_idx] > M_PI) state[angle_idx] -= 2*M_PI;
            while (state[angle_idx] < -M_PI) state[angle_idx] += 2*M_PI;
        }
        
        times.push_back(t + dt);
        states.push_back(state);
        
        // Check ground contact
        if (state[2] >= 0.0) break;
    }
    
    return {times, states};
}