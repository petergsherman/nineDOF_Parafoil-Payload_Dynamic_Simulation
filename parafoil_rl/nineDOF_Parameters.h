#ifndef NINEDOF_PARAMETERS_H
#define NINEDOF_PARAMETERS_H

#include <vector>

// ============================================================================
// STATE VECTOR LAYOUT (18 states total)
// ============================================================================
// States 0-2:   xg, yg, zg          - Gimbal position in inertial frame (m)
// States 3-5:   phip, thetap, psip  - Parafoil Euler angles (rad)
// States 6-8:   phic, thetac, psic  - Cradle Euler angles (rad)
// States 9-11:  ug, vg, wg          - Gimbal velocity in parafoil frame (m/s)
// States 12-14: pp, qp, rp          - Parafoil angular rates (rad/s)
// States 15-17: pc, qc, rc          - Cradle angular rates (rad/s)
// ============================================================================

struct SystemParameters {
    // Masses and Geometry
    double m_parafoil = 8.99;      // Parafoil mass (kg)
    double m_cradle = 90.0;        // Cradle mass (kg)
    double A_parafoil = 27.0;      // Parafoil area (m^2)
    double A_cradle = 0.4337;      // Cradle area (m^2)
    double cbar_parafoil = 3.87;   // Parafoil mean chord (m)
    double bbar_parafoil = 6.99;   // Parafoil span (m)
    double dbar = 1.0;             // Brake deflection reference
    double deadband = 0.0;         // Control deadband
    double CD_cradle = 1.0;        // Cradle drag coefficient

    // Parafoil Inertia (kg-m^2)
    double PIXX = 74.56;
    double PIXY = 0.0;
    double PIXZ = 0.0;
    double PIYY = 14.62;
    double PIYZ = 0.0;
    double PIZZ = 82.8;
    
    // Cradle Inertia (kg-m^2)
    double CIXX = 9.378;
    double CIXY = 0.0;
    double CIXZ = 0.0;
    double CIYY = 6.0518;
    double CIYZ = 0.0;
    double CIZZ = 6.2401;

    // Geometry vectors (m)
    double SLCRADLE = 0.0;
    double BLCRADLE = 0.0;
    double WLCRADLE = 0.47;
    
    double SLPARAFOIL = 0.0;
    double BLPARAFOIL = 0.0;
    double WLPARAFOIL = -7.622;
    
    double SLPR = 0.0;
    double BLPR = 0.0;
    double WLPR = 0.0;
    
    double SLRAP = 0.0;
    double BLRAP = 0.0;
    double WLRAP = 0.0;
    
    double SLPMP = 0.0;
    double BLPMP = 0.0;
    double WLPMP = 7.622;

    // Apparent Mass and Inertia Coefficients
    double PMASSA = 1.05;
    double PMASSB = 6.46;
    double PMASSC = 31.78;
    double PMASSH = 0.0;
    double PMASSP = 18.36;
    double PMASSQ = 26.5;
    double PMASSR = 7.104;
    
    // Gimbal Properties
    double KGIMBAL = 16.244;   // Rotational Stiffness (N-m/rad)
    double CGIMBAL = 1.3537;   // Rotational Damping (N-m/(rad/s))
    
    // Control parameters
    double NOM_INCIDENCE = -0.055;  // Nominal incidence angle (rad)
    double NOM_BRAKE = 0.5;         // Nominal brake deflection (m)
    double TAU = 0.6;               // Brake time constant (s)
    
    // Aerodynamic coefficients (single-point model)
    double CM0 = 0.0;
    double CMQ = -1.0;
    double CMDS = 0.0;
    double CYBETA = -1.0;
    double CNBETA = 0.0;
    double CLBETA = 0.0;
    double CLP = 0.112;
    double CLR = 0.253;
    double CLDA = 0.0;
    double CNP = -0.050;
    double CNR = -0.212;
    double CND1 = 0.0;

    // Aerodynamic tables
    std::vector<double> SIGTAB = {0.2500, 1.5000};
    std::vector<double> CD0TAB = {0.1373, 0.0351};
    std::vector<double> CDA2TAB = {0.0977, 3.9642};
    std::vector<double> CL0TAB = {-0.2636, 0.1313};
    std::vector<double> CLATAB = {2.8139, 1.9825};
    
    // AOA table
    std::vector<double> AOATAB = {0.1800, 0.2500, 0.3200};
    std::vector<double> CND2TAB = {0.0280, 0.0350, 0.0700};
    
    // Physical constants
    double GRAVITY = 9.81;  // m/s^2
};

struct AtmosphereParameters {
    double DEN = 1.22566;      // Air density (kg/m^3)
    double VXWIND = 0.0;       // Wind velocity X (m/s)
    double VYWIND = 0.0;       // Wind velocity Y (m/s)
    double VZWIND = 0.0;       // Wind velocity Z (m/s)
    
    virtual void update(double t, double altitude) {}
};

#endif // NINEDOF_PARAMETERS_H