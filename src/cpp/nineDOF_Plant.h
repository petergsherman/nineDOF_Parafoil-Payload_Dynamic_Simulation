#ifndef NINEDOF_PLANT_H
#define NINEDOF_PLANT_H

#include "nineDOF_Parameters.h"
#include "nineDOF_Transform.h"
#include <vector>
#include <array>
#include <functional>

using State = std::array<double, 18>;
using ControlFunc = std::function<std::array<double, 3>(double, const State&)>;

class Plant {
private:
    SystemParameters params;
    AtmosphereParameters* atm;
    
    // Inertia and mass matrices
    Matrix3x3 I_P;   // Parafoil inertia
    Matrix3x3 I_C;   // Cradle inertia
    Matrix3x3 I_AM;  // Apparent mass
    Matrix3x3 I_AI;  // Apparent inertia
    Matrix3x3 I_H;   // Spanwise camber matrix
    
    void populateInertias();
    Matrix3x3 makeT_PPI(double incidence, double nominalIncidence);
    Matrix3x3 skew(const Vector3& v);
    
    void interpolateAeroTables(double sig1, double& cd0, double& cda2, double& cl0, double& cla);
    double interpolateCND2(double alpha);
    
    void computeAerodynamics(const State& state, double deltaL, double deltaR, 
                            double incidence, Vector3& C_FAp_P, Vector3& C_MAp_P, 
                            Vector3& C_FAc_C);
    
    State computeDerivatives(const State& state, double deltaL, double deltaR, double incidence);
    State rk4_step(const State& state, double dt, double deltaL, double deltaR, double incidence);
    
public:
    Plant(const SystemParameters& parameters, AtmosphereParameters* atmosphere);
    
    std::pair<std::vector<double>, std::vector<State>> 
        run_simulation(const State& state0, double t_final, double dt, ControlFunc control_func = nullptr);
};

#endif // NINEDOF_PLANT_H