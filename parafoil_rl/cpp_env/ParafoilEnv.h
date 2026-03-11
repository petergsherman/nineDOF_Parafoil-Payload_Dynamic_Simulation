#pragma once
// =============================================================================
// ParafoilEnv.h
// Reinforcement Learning environment wrapper around the 9DOF parafoil plant.
//
// Architecture note:
//   - This class OWNS the episode state and advances it one RL timestep at a time.
//   - The underlying Plant (nineDOF_Plant) does the physics; this class manages
//     the RL interface: reset, step, observation, reward, done.
//   - Designed to be bound to Python via pybind11 (see parafoil_bindings.cpp).
//
// State vector layout (18 states):
//   0-2:   xg, yg, zg          - Gimbal position in inertial frame (m)
//   3-5:   phip, thetap, psip  - Parafoil Euler angles (rad)
//   6-8:   phic, thetac, psic  - Cradle Euler angles (rad)
//   9-11:  ug, vg, wg          - Gimbal velocity in parafoil frame (m/s)
//   12-14: pp, qp, rp          - Parafoil angular rates (rad/s)
//   15-17: pc, qc, rc          - Cradle angular rates (rad/s)
//
// Action: [deltaL, deltaR] - two brake commands in [0, 0.94]
//   Incidence is held at nominal (0.0 perturbation) during RL training.
//   This gives the policy a simple, direct, physically meaningful action space.
//
// Observation vector (12 elements - see getObservation() for details):
//   [dx_norm, dy_norm, alt_norm, heading, heading_error,
//    speed, vz, phi_p, theta_p, p_p, q_p, r_p]
// =============================================================================

#ifndef PARAFOIL_ENV_H
#define PARAFOIL_ENV_H

#include "nineDOF_Plant.h"
#include "nineDOF_Parameters.h"
#include <array>
#include <vector>
#include <random>
#include <string>
#include <tuple>

// ----------------------------------------------------------------------------
// Domain randomization configuration
// Edit these ranges to tune curriculum difficulty.
// ----------------------------------------------------------------------------
struct DomainRandomConfig {
    // Initial position spread around target (m)
    double pos_x_range   = 600.0;   // +/- x spread
    double pos_y_range   = 600.0;   // +/- y spread
    double alt_min       = 400.0;   // Minimum initial altitude (m AGL)
    double alt_max       = 700.0;   // Maximum initial altitude (m AGL)

    // Initial heading spread (rad)
    double heading_range = M_PI;    // Full 360-degree randomization

    // Initial velocity (m/s) - small perturbation around nominal trim
    double u_nominal     = 10.0;
    double u_noise       = 1.0;
    double w_nominal     = -0.5;
    double w_noise       = 0.2;

    // Wind (static atmosphere training)
    double wind_x_range  = 0.0;     // Zero wind for initial training
    double wind_y_range  = 0.0;

    // Small attitude perturbations (rad)
    double theta_noise   = 0.05;
    double phi_noise     = 0.02;
};

// ----------------------------------------------------------------------------
// Episode info struct returned alongside step()
// ----------------------------------------------------------------------------
struct StepInfo {
    double distance_to_target;   // 2D horizontal distance (m)
    double altitude;             // Current altitude AGL (m)
    double heading;              // Current heading (rad)
    double airspeed;             // Airspeed magnitude (m/s)
    bool   hit_ground;           // True if z >= 0
    bool   diverged;             // True if state became invalid
    int    step_count;           // Steps taken this episode
    double landing_error;        // Final 2D landing error (m), set on done
};

// Observation size - update OBS_SIZE if you change getObservation()
static constexpr int OBS_SIZE = 12;
static constexpr int ACT_SIZE = 2;   // [deltaL, deltaR]

class ParafoilEnv {
public:
    // -------------------------------------------------------------------------
    // Constructor
    // params       - system parameters (same struct as your existing plant)
    // target_x/y   - landing target in inertial frame (m)
    // dt_physics   - physics integration timestep (s), default 0.01
    // dt_action    - time between RL actions (s), default 0.1
    //   The env will run (dt_action / dt_physics) physics steps per RL step.
    // -------------------------------------------------------------------------
    ParafoilEnv(const SystemParameters& params,
                double target_x = 0.0,
                double target_y = 0.0,
                double dt_physics = 0.01,
                double dt_action  = 0.1);

    // -------------------------------------------------------------------------
    // reset() - start a new episode
    // seed        - RNG seed (-1 = use internal counter)
    // fixed_state - if non-empty, use this exact state (overrides randomization)
    // Returns:    observation vector
    // -------------------------------------------------------------------------
    std::array<double, OBS_SIZE> reset(int seed = -1,
                                       const State* fixed_state = nullptr);

    // -------------------------------------------------------------------------
    // step() - advance one RL timestep
    // action - [deltaL, deltaR] in [0.0, 0.94]
    // Returns: (observation, reward, done, info)
    // -------------------------------------------------------------------------
    std::tuple<std::array<double, OBS_SIZE>, double, bool, StepInfo>
        step(const std::array<double, ACT_SIZE>& action);

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------
    std::array<double, OBS_SIZE> getObservation() const;
    bool    isDone()   const;
    double  getTime()  const { return current_time_; }
    State   getState() const { return current_state_; }
    StepInfo getInfo() const { return last_info_; }

    // -------------------------------------------------------------------------
    // Configuration setters - call before reset()
    // -------------------------------------------------------------------------
    void setTarget(double x, double y) { target_x_ = x; target_y_ = y; }
    void setDomainRandomConfig(const DomainRandomConfig& cfg) { dr_cfg_ = cfg; }
    void setMaxEpisodeTime(double t) { max_episode_time_ = t; }

    // Static atmosphere wind (for domain randomization override)
    void setWind(double vx, double vy, double vz = 0.0);

    // Observation normalization bounds (public so Python wrapper can build gym spaces)
    static constexpr double OBS_POS_NORM   = 1000.0;  // m
    static constexpr double OBS_ALT_NORM   = 800.0;   // m
    static constexpr double OBS_VEL_NORM   = 20.0;    // m/s
    static constexpr double OBS_RATE_NORM  = 1.0;     // rad/s
    static constexpr double OBS_ANGLE_NORM = M_PI;    // rad

private:
    // Plant and atmosphere owned by env
    SystemParameters   params_;
    AtmosphereParameters atm_;
    Plant              plant_;

    // Target landing point
    double target_x_, target_y_;

    // Simulation parameters
    double dt_physics_;
    double dt_action_;
    int    steps_per_action_;
    double max_episode_time_ = 1200.0;  // 20 min hard timeout

    // Episode state
    State  current_state_;
    double current_time_;
    int    step_count_;
    bool   episode_done_;
    StepInfo last_info_;

    // Previous values for reward shaping
    double prev_distance_;
    double prev_control_effort_;

    // Domain randomization config
    DomainRandomConfig dr_cfg_;

    // RNG
    std::mt19937 rng_;
    int episode_counter_ = 0;

    // Internal helpers
    double computeReward(const std::array<double, ACT_SIZE>& action,
                         bool done, bool diverged);
    void   wrapAngles(State& state);
    bool   checkDivergence(const State& state) const;
    double horizontalDistance() const;
    double getCurrentHeading() const;
};

#endif // PARAFOIL_ENV_H
