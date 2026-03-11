// =============================================================================
// ParafoilEnv.cpp
// RL environment implementation - wraps the 9DOF Plant for Gymnasium/SB3 use.
// =============================================================================

#include "ParafoilEnv.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Clamp action to valid physical range
static inline double clampAction(double val, double lo, double hi) {
    return std::max(lo, std::min(hi, val));
}

// Angle difference wrapped to [-pi, pi]
static inline double angleDiff(double a, double b) {
    double d = a - b;
    while (d >  M_PI) d -= 2.0 * M_PI;
    while (d < -M_PI) d += 2.0 * M_PI;
    return d;
}

// =============================================================================
// Constructor
// =============================================================================
ParafoilEnv::ParafoilEnv(const SystemParameters& params,
                         double target_x, double target_y,
                         double dt_physics, double dt_action)
    : params_(params),
      plant_(params, &atm_),
      target_x_(target_x),
      target_y_(target_y),
      dt_physics_(dt_physics),
      dt_action_(dt_action),
      rng_(std::random_device{}())
{
    steps_per_action_ = std::max(1, static_cast<int>(std::round(dt_action_ / dt_physics_)));
    current_state_.fill(0.0);
    current_time_ = 0.0;
    step_count_   = 0;
    episode_done_ = true;
    prev_distance_ = 0.0;
}

// =============================================================================
// reset()
// =============================================================================
std::array<double, OBS_SIZE> ParafoilEnv::reset(int seed, const State* fixed_state) {
    // Seed RNG
    if (seed >= 0) {
        rng_.seed(static_cast<unsigned>(seed));
    } else {
        rng_.seed(static_cast<unsigned>(episode_counter_ * 1337 + 42));
    }
    episode_counter_++;

    if (fixed_state != nullptr) {
        // Use provided state exactly (for deterministic evaluation)
        current_state_ = *fixed_state;
    } else {
        // ------------------------------------------------------------------
        // Domain randomization
        // All randomization is concentrated here - easy to tune later.
        // ------------------------------------------------------------------
        auto uniform = [&](double lo, double hi) -> double {
            return std::uniform_real_distribution<double>(lo, hi)(rng_);
        };

        // 1. Randomize initial position around target
        double x_offset = uniform(-dr_cfg_.pos_x_range, dr_cfg_.pos_x_range);
        double y_offset = uniform(-dr_cfg_.pos_y_range, dr_cfg_.pos_y_range);
        double altitude = uniform(dr_cfg_.alt_min, dr_cfg_.alt_max);

        current_state_[0] = target_x_ + x_offset;
        current_state_[1] = target_y_ + y_offset;
        current_state_[2] = -altitude;   // z is negative-down (NED)

        // 2. Randomize heading (psi_p and psi_c together)
        double heading = uniform(-dr_cfg_.heading_range, dr_cfg_.heading_range);
        current_state_[3] = uniform(-dr_cfg_.phi_noise,   dr_cfg_.phi_noise);    // phi_p
        current_state_[4] = uniform(-dr_cfg_.theta_noise, dr_cfg_.theta_noise);  // theta_p
        current_state_[5] = heading;                                               // psi_p
        current_state_[6] = uniform(-dr_cfg_.phi_noise,   dr_cfg_.phi_noise);    // phi_c
        current_state_[7] = uniform(-dr_cfg_.theta_noise, dr_cfg_.theta_noise);  // theta_c
        current_state_[8] = heading;                                               // psi_c (same as parafoil initially)

        // 3. Nominal trim velocity with small noise
        current_state_[9]  = dr_cfg_.u_nominal + uniform(-dr_cfg_.u_noise, dr_cfg_.u_noise);
        current_state_[10] = 0.0;
        current_state_[11] = dr_cfg_.w_nominal + uniform(-dr_cfg_.w_noise, dr_cfg_.w_noise);

        // 4. Zero angular rates
        for (int i = 12; i < 18; i++) current_state_[i] = 0.0;

        // 5. Wind randomization (zero for static atmosphere training)
        atm_.VXWIND = uniform(-dr_cfg_.wind_x_range, dr_cfg_.wind_x_range);
        atm_.VYWIND = uniform(-dr_cfg_.wind_y_range, dr_cfg_.wind_y_range);
        atm_.VZWIND = 0.0;
    }

    current_time_  = 0.0;
    step_count_    = 0;
    episode_done_  = false;
    prev_distance_ = horizontalDistance();
    prev_control_effort_ = 0.0;

    last_info_ = {};
    last_info_.distance_to_target = prev_distance_;
    last_info_.altitude           = -current_state_[2];

    return getObservation();
}

// =============================================================================
// step()
// =============================================================================
std::tuple<std::array<double, OBS_SIZE>, double, bool, StepInfo>
ParafoilEnv::step(const std::array<double, ACT_SIZE>& action) {
    if (episode_done_) {
        throw std::runtime_error("ParafoilEnv::step() called after episode done. Call reset() first.");
    }

    // Clamp actions to physical limits [0, 0.94]
    double deltaL    = clampAction(action[0], 0.0, 0.94);
    double deltaR    = clampAction(action[1], 0.0, 0.94);
    double incidence = 0.0;   // Fixed at nominal during RL training

    bool diverged = false;

    // Run physics sub-steps at dt_physics for one RL action step
    for (int sub = 0; sub < steps_per_action_; ++sub) {
        current_state_ = plant_.rk4_step(current_state_, dt_physics_, deltaL, deltaR, incidence);
        wrapAngles(current_state_);

        if (checkDivergence(current_state_)) {
            diverged = true;
            break;
        }

        current_time_ += dt_physics_;

        // Ground contact check
        if (current_state_[2] >= 0.0) {
            current_state_[2] = 0.0;  // Clamp to ground
            break;
        }
    }

    step_count_++;
    double altitude  = -current_state_[2];
    bool   hit_ground = (current_state_[2] >= 0.0);
    bool   timeout    = (current_time_ >= max_episode_time_);
    bool   done       = hit_ground || diverged || timeout;

    // Build info
    StepInfo info;
    info.distance_to_target = horizontalDistance();
    info.altitude           = altitude;
    info.heading            = getCurrentHeading();
    info.airspeed           = std::sqrt(current_state_[9]*current_state_[9] +
                                        current_state_[10]*current_state_[10] +
                                        current_state_[11]*current_state_[11]);
    info.hit_ground         = hit_ground;
    info.diverged           = diverged;
    info.step_count         = step_count_;
    info.landing_error      = done ? info.distance_to_target : -1.0;

    // Compute reward
    double reward = computeReward(action, done, diverged);

    episode_done_ = done;
    prev_distance_ = info.distance_to_target;
    last_info_ = info;

    return {getObservation(), reward, done, info};
}

// =============================================================================
// getObservation()
//
// 12-element observation vector designed for end-to-end control.
// All terms normalized to roughly [-1, 1] for stable NN training.
//
// [0]  dx_norm          - Relative x to target, normalized by OBS_POS_NORM
// [1]  dy_norm          - Relative y to target, normalized by OBS_POS_NORM
// [2]  alt_norm         - Altitude, normalized by OBS_ALT_NORM
// [3]  sin(psi_p)       - Parafoil heading (sin component)
// [4]  cos(psi_p)       - Parafoil heading (cos component)
// [5]  sin(heading_err) - Heading error to target (sin component)
// [6]  cos(heading_err) - Heading error to target (cos component)
// [7]  speed_norm       - Horizontal airspeed magnitude, normalized
// [8]  vz_norm          - Vertical velocity (descent rate), normalized
// [9]  phi_p_norm       - Parafoil roll angle, normalized
// [10] theta_p_norm     - Parafoil pitch angle, normalized
// [11] rp_norm          - Parafoil yaw rate, normalized
//
// Design rationale:
//   - Relative position tells the policy WHERE to go.
//   - sin/cos encoding avoids angle discontinuities at +/-pi.
//   - Heading error gives a direct steering signal.
//   - Speed and descent rate indicate energy state.
//   - Roll/pitch/yaw-rate give stability feedback.
//   - Angular rates of cradle omitted for brevity - add if policy needs them.
// =============================================================================
std::array<double, OBS_SIZE> ParafoilEnv::getObservation() const {
    const State& s = current_state_;

    double dx  = target_x_ - s[0];
    double dy  = target_y_ - s[1];
    double alt = -s[2];

    double psi_p = s[5];
    double heading_to_target = std::atan2(dy, dx);
    double heading_err       = angleDiff(heading_to_target, psi_p);

    double speed = std::sqrt(s[9]*s[9] + s[10]*s[10]);
    double vz    = s[11];  // Positive = descending in body frame

    std::array<double, OBS_SIZE> obs;
    obs[0]  = std::max(-1.0, std::min(1.0, dx  / OBS_POS_NORM));
    obs[1]  = std::max(-1.0, std::min(1.0, dy  / OBS_POS_NORM));
    obs[2]  = std::max(0.0,  std::min(1.0, alt / OBS_ALT_NORM));
    obs[3]  = std::sin(psi_p);
    obs[4]  = std::cos(psi_p);
    obs[5]  = std::sin(heading_err);
    obs[6]  = std::cos(heading_err);
    obs[7]  = std::max(-1.0, std::min(1.0, speed / OBS_VEL_NORM));
    obs[8]  = std::max(-1.0, std::min(1.0, vz    / OBS_VEL_NORM));
    obs[9]  = std::max(-1.0, std::min(1.0, s[3]  / OBS_ANGLE_NORM));  // phi_p
    obs[10] = std::max(-1.0, std::min(1.0, s[4]  / OBS_ANGLE_NORM));  // theta_p
    obs[11] = std::max(-1.0, std::min(1.0, s[14] / OBS_RATE_NORM));   // r_p (yaw rate)

    return obs;
}

// =============================================================================
// computeReward()
//
// Reward design rationale:
//   The primary objective is minimizing landing distance from the target.
//   To help exploration before landing:
//     - Dense shaping: reward proportional to distance reduction per step.
//     - Alive bonus: small positive reward to encourage longer episodes.
//     - Control penalty: small penalty on asymmetric brake usage to avoid
//       excessive oscillation.
//     - Terminal: large bonus/penalty on landing, scaled by proximity.
//     - Divergence: large penalty to avoid unphysical states.
//
//   All reward components are tuned to be in a similar magnitude range.
// =============================================================================
double ParafoilEnv::computeReward(const std::array<double, ACT_SIZE>& action,
                                  bool done, bool diverged) {
    double reward = 0.0;

    double dist = horizontalDistance();

    // 1. Progress reward: reward getting closer to target
    //    Scale by dt_action so reward is roughly rate-independent
    double dist_reduction = prev_distance_ - dist;
    reward += 2.0 * dist_reduction;

    // 2. Alive bonus: encourages stable descent (per step)
    reward += 0.01;

    // 3. Control effort penalty: penalize differential (yaw-inducing) brake
    //    symmetric brake is fine, asymmetric causes oscillation
    double dA = std::abs(action[1] - action[0]);
    reward -= 0.05 * dA;

    // 4. Terminal rewards
    if (done) {
        if (diverged) {
            // Heavy penalty for numerical divergence
            reward -= 200.0;
        } else {
            // Landing bonus/penalty scaled by accuracy
            // dist=0   -> +200, dist=50m -> +150, dist=500m -> -100
            double landing_bonus = 200.0 - 0.6 * dist;
            reward += landing_bonus;

            // Extra bonus for very accurate landing
            if (dist < 25.0)  reward += 100.0;
            if (dist < 10.0)  reward += 200.0;
        }
    }

    return reward;
}

// =============================================================================
// Helpers
// =============================================================================
bool ParafoilEnv::isDone() const {
    return episode_done_;
}

void ParafoilEnv::wrapAngles(State& state) {
    // Wrap Euler angles to [-pi, pi]
    for (int idx : {3, 4, 5, 6, 7, 8}) {
        while (state[idx] >  M_PI) state[idx] -= 2.0 * M_PI;
        while (state[idx] < -M_PI) state[idx] += 2.0 * M_PI;
    }
}

bool ParafoilEnv::checkDivergence(const State& state) const {
    // Check for NaN/Inf
    for (double v : state) {
        if (std::isnan(v) || std::isinf(v)) return true;
    }
    // Check for physically unreasonable velocities
    double speed = std::sqrt(state[9]*state[9] + state[10]*state[10] + state[11]*state[11]);
    if (speed > 200.0) return true;
    // Check for extreme position (escaped envelope)
    if (std::abs(state[0]) > 50000.0 || std::abs(state[1]) > 50000.0) return true;
    return false;
}

double ParafoilEnv::horizontalDistance() const {
    double dx = target_x_ - current_state_[0];
    double dy = target_y_ - current_state_[1];
    return std::sqrt(dx*dx + dy*dy);
}

double ParafoilEnv::getCurrentHeading() const {
    return current_state_[5];  // psi_p
}

void ParafoilEnv::setWind(double vx, double vy, double vz) {
    atm_.VXWIND = vx;
    atm_.VYWIND = vy;
    atm_.VZWIND = vz;
}
