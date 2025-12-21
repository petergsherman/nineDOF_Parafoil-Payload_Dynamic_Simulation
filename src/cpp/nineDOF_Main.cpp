#include "nineDOF_Plant.h"
#include "nineDOF_Parameters.h"
#include <iostream>
#include <fstream>
#include <iomanip>

// Example control function
std::array<double, 3> testController(double t, const State& state) {
    // Returns [deltaL, deltaR, incidence]
    return {0.0, 0.0, 0.0};  // No control
}

// Simple visualization - save to CSV
void saveTrajectory(const std::vector<double>& times, 
                   const std::vector<State>& states,
                   const std::string& filename = "trajectory.csv") {
    std::ofstream file(filename);
    
    file << std::fixed << std::setprecision(6);
    file << "time,x,y,z,phi_p,theta_p,psi_p,phi_c,theta_c,psi_c,u,v,w,p_p,q_p,r_p,p_c,q_c,r_c\n";
    
    for (size_t i = 0; i < times.size(); i++) {
        file << times[i];
        for (int j = 0; j < 18; j++) {
            file << "," << states[i][j];
        }
        file << "\n";
    }
    
    file.close();
    std::cout << "Trajectory saved to " << filename << std::endl;
}

int main() {
    // Create parameters
    SystemParameters params;
    AtmosphereParameters atm;
    
    // Create simulator
    Plant sim(params, &atm);
    
    // Initial state: hovering at origin with small perturbation
    State state0 = {0};
    state0[0] = 0.0;    state0[1] = 0.0;    state0[2] = -1000.0;  // Position: 1000m altitude
    state0[3] = 0.0;    state0[4] = 0.1;    state0[5] = 0.0;      // Parafoil angles
    state0[6] = 0.0;    state0[7] = 0.1;    state0[8] = 0.0;      // Cradle angles
    state0[9] = 10.0;   state0[10] = 0.0;   state0[11] = -0.5;    // Forward velocity
    state0[12] = 0.0;   state0[13] = 0.0;   state0[14] = 0.0;     // Parafoil rates
    state0[15] = 0.0;   state0[16] = 0.0;   state0[17] = 0.0;     // Cradle rates
    
    // Run simulation
    std::cout << "Running parafoil-payload simulation..." << std::endl;
    double t_final = 1000.0;  // seconds
    double dt = 0.1;          // seconds
    
    // Create control function
    ControlFunc control = testController;
    
    auto [times, states] = sim.run_simulation(state0, t_final, dt, control);
    
    std::cout << "Simulation complete: " << times.size() << " time steps" << std::endl;
    
    const State& final_state = states.back();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Final position: x=" << final_state[0] 
              << ", y=" << final_state[1] 
              << ", z=" << final_state[2] << std::endl;
    std::cout << "Final velocity: u=" << final_state[9] 
              << ", v=" << final_state[10] 
              << ", w=" << final_state[11] << std::endl;
    
    // Save trajectory
    saveTrajectory(times, states);
    
    return 0;
}