"""
PID Tuning Script for Parafoil Controller

This script provides both manual and automated methods for tuning PID gains.
It assumes you have a simulation object with a run_simulation method.

Usage:
    python pid_tuning.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from nineDOF_Control import PIDHeadingController, make_control_function

# ============================================================================
# CONFIGURATION
# ============================================================================

from nineDOF_Plant import plant
from nineDOF_Parameters import systemParameters
from nineDOF_Atmosphere import staticAtmosphere

sim = plant(systemParameters(), staticAtmosphere())

# Initial state for simulation
state0 = np.zeros(18)
state0[0:3] = [0.0, 0.0, -1000.0]  # Position: 100m altitude
state0[3:6] = [0.0, 0.1, 0.0]     # Parafoil angles: small pitch
state0[6:9] = [0.0, 0.1, 0.0]     # Cradle angles: small pitch
state0[9:12] = [10.0, 0.0, -0.5]  # Forward velocity

# Simulation parameters
T_FINAL = 100.0  # Total simulation time (seconds)
DT = 0.01  # Time step (seconds)
TARGET_LOCATION = (1000.0, 1000.0)  # Target landing coordinates (x, y)

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def evaluate_controller(kp, ki, kd, sim, state0, target, t_final=T_FINAL, dt=DT, verbose=False):
    """
    Evaluate PID controller performance with given gains.
    
    Returns a cost that combines:
    - Final distance to target (primary objective)
    - Average distance during flight
    - Control effort (smoothness)
    - Settling time
    """
    
    # Create controller with given gains
    controller = PIDHeadingController(
        targetLandingLocation=target,
        kp=kp, ki=ki, kd=kd,
        dt=dt
    )
    
    control_func = make_control_function(controller)
    
    try:
        # Run simulation
        times, states = sim.run_simulation(state0, t_final, dt, control_func)
        
        # Extract positions
        x_positions = states[:, 0]
        y_positions = states[:, 1]
        
        # Calculate distances to target over time
        distances = np.sqrt((x_positions - target[0])**2 + (y_positions - target[1])**2)
        
        # Performance metrics
        final_distance = distances[-1]
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        
        # Control effort (measure of control smoothness)
        controls = []
        for i, state in enumerate(states):
            deltaL, deltaR, _ = controller.computeControl(state, times[i])
            controls.append(abs(deltaL) + abs(deltaR))
        avg_control = np.mean(controls)
        control_variance = np.var(controls)
        
        # Settling time (time to get within 20m of target and stay there)
        settling_threshold = 20.0
        settled_idx = np.where(distances < settling_threshold)[0]
        if len(settled_idx) > 0:
            settling_time = times[settled_idx[0]]
        else:
            settling_time = t_final  # Never settled
        
        # Combined cost function (weights can be tuned)
        cost = (
            10.0 * final_distance +          # Heavily penalize final miss distance
            1.0 * avg_distance +              # Penalize average distance
            0.1 * avg_control +               # Small penalty for control effort
            0.5 * control_variance +          # Penalize erratic control
            0.5 * settling_time               # Penalize slow settling
        )
        
        if verbose:
            print(f"  kp={kp:.3f}, ki={ki:.3f}, kd={kd:.3f}")
            print(f"  Final distance: {final_distance:.2f}m")
            print(f"  Avg distance: {avg_distance:.2f}m")
            print(f"  Settling time: {settling_time:.2f}s")
            print(f"  Cost: {cost:.2f}")
        
        return cost, {
            'final_distance': final_distance,
            'avg_distance': avg_distance,
            'settling_time': settling_time,
            'avg_control': avg_control,
            'times': times,
            'states': states,
            'distances': distances
        }
        
    except Exception as e:
        print(f"Simulation failed with kp={kp}, ki={ki}, kd={kd}: {e}")
        return 1e10, None  # Return very high cost for failed simulations

# ============================================================================
# MANUAL TUNING HELPER
# ============================================================================

def manual_tuning_test(sim, state0, target, kp_range, ki_range, kd_range):
    """
    Test a range of PID values and visualize results.
    Good for understanding the effect of each parameter.
    """
    
    print("\n" + "="*70)
    print("MANUAL PID TUNING - Testing parameter ranges")
    print("="*70)
    
    results = []
    
    # Test Kp variations (with Ki=0, Kd=0)
    print("\n--- Testing Kp (Proportional Gain) ---")
    for kp in kp_range:
        cost, metrics = evaluate_controller(kp, 0.0, 0.0, sim, state0, target, verbose=True)
        results.append(('P', kp, 0.0, 0.0, cost, metrics))
    
    # Test Kp + Kd variations (with Ki=0)
    print("\n--- Testing Kd (Derivative Gain) ---")
    best_kp = kp_range[len(kp_range)//2]  # Use middle Kp value
    for kd in kd_range:
        cost, metrics = evaluate_controller(best_kp, 0.0, kd, sim, state0, target, verbose=True)
        results.append(('PD', best_kp, 0.0, kd, cost, metrics))
    
    # Test full PID
    print("\n--- Testing Ki (Integral Gain) ---")
    best_kd = kd_range[len(kd_range)//2]  # Use middle Kd value
    for ki in ki_range:
        cost, metrics = evaluate_controller(best_kp, ki, best_kd, sim, state0, target, verbose=True)
        results.append(('PID', best_kp, ki, best_kd, cost, metrics))
    
    return results

# ============================================================================
# AUTOMATED OPTIMIZATION
# ============================================================================

def optimize_pid_gradient(sim, state0, target, initial_guess=(0.2, 0.05, 0.3)):
    """
    Use gradient-based optimization (Nelder-Mead) to find optimal PID gains.
    Fast but may find local minima.
    """
    
    print("\n" + "="*70)
    print("GRADIENT-BASED OPTIMIZATION (Nelder-Mead)")
    print("="*70)
    
    def cost_function(params):
        kp, ki, kd = params
        # Ensure positive gains
        kp, ki, kd = abs(kp), abs(ki), abs(kd)
        cost, _ = evaluate_controller(kp, ki, kd, sim, state0, target)
        print(f"Testing: kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f} -> Cost={cost:.2f}")
        return cost
    
    result = minimize(
        cost_function,
        x0=initial_guess,
        method='Nelder-Mead',
        options={'maxiter': 100, 'disp': True}
    )
    
    optimal_kp, optimal_ki, optimal_kd = abs(result.x[0]), abs(result.x[1]), abs(result.x[2])
    
    print(f"\n✓ Optimal PID gains found:")
    print(f"  Kp = {optimal_kp:.4f}")
    print(f"  Ki = {optimal_ki:.4f}")
    print(f"  Kd = {optimal_kd:.4f}")
    
    return optimal_kp, optimal_ki, optimal_kd

def optimize_pid_global(sim, state0, target):
    """
    Use global optimization (Differential Evolution) to find optimal PID gains.
    Slower but more thorough - explores the entire parameter space.
    """
    
    print("\n" + "="*70)
    print("GLOBAL OPTIMIZATION (Differential Evolution)")
    print("="*70)
    
    def cost_function(params):
        kp, ki, kd = params
        cost, _ = evaluate_controller(kp, ki, kd, sim, state0, target)
        return cost
    
    # Define bounds for PID gains
    bounds = [
        (0.01, 2.0),   # Kp bounds
        (0.0, 0.5),    # Ki bounds
        (0.0, 2.0)     # Kd bounds
    ]
    
    result = differential_evolution(
        cost_function,
        bounds=bounds,
        maxiter=30,
        popsize=10,
        disp=True,
        workers=1  # Set to -1 for parallel processing if available
    )
    
    optimal_kp, optimal_ki, optimal_kd = result.x
    
    print(f"\n✓ Optimal PID gains found:")
    print(f"  Kp = {optimal_kp:.4f}")
    print(f"  Ki = {optimal_ki:.4f}")
    print(f"  Kd = {optimal_kd:.4f}")
    
    return optimal_kp, optimal_ki, optimal_kd

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison(results_list, labels, target):
    """
    Plot trajectories and performance metrics for different PID configurations.
    
    Args:
        results_list: List of (times, states, distances) tuples
        labels: List of labels for each result
        target: Target location (x, y)
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: 2D Trajectory
    ax = axes[0, 0]
    for (times, states, distances), label in zip(results_list, labels):
        ax.plot(states[:, 0], states[:, 1], label=label, linewidth=2)
    ax.plot(target[0], target[1], 'r*', markersize=20, label='Target')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Flight Trajectory (Top View)')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # Plot 2: Distance to Target vs Time
    ax = axes[0, 1]
    for (times, states, distances), label in zip(results_list, labels):
        ax.plot(times, distances, label=label, linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance to Target (m)')
    ax.set_title('Distance to Target Over Time')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Altitude vs Time
    ax = axes[1, 0]
    for (times, states, distances), label in zip(results_list, labels):
        altitude = -states[:, 2]  # Convert to positive altitude
        ax.plot(times, altitude, label=label, linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude Profile')
    ax.legend()
    ax.grid(True)
    
    # Plot 4: Heading Error vs Time
    ax = axes[1, 1]
    for (times, states, distances), label in zip(results_list, labels):
        heading_errors = []
        for state in states:
            current_pos = np.array([state[0], state[1]])
            current_heading = state[5]
            to_target = target - current_pos
            desired_heading = np.arctan2(to_target[1], to_target[0])
            heading_error = desired_heading - current_heading
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            heading_errors.append(np.rad2deg(heading_error))
        ax.plot(times, heading_errors, label=label, linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading Error (degrees)')
    ax.set_title('Heading Error Over Time')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('pid_tuning_results.png', dpi=150)
    plt.show()
    
    print("\n✓ Plots saved to 'pid_tuning_results.png'")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main tuning workflow.
    Uncomment the sections you want to run.
    """
    
    # TODO: Replace these with your actual simulation setup
    # from your_plant_module import YourPlant
    # sim = YourPlant()
    # state0 = np.array([100, 100, -300, 0, 0, 0, ...])  # Start 100m away from target
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║           PID CONTROLLER TUNING FOR PARAFOIL SYSTEM               ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Uncomment the method you want to use:
    
    # -------------------------------------------------------------------------
    # METHOD 1: Manual exploration (understand parameter effects)
    #-------------------------------------------------------------------------
    # kp_range = np.linspace(0.1, 1.0, 5)
    # ki_range = np.linspace(0.0, 0.2, 5)
    # kd_range = np.linspace(0.1, 1.0, 5)
    # results = manual_tuning_test(sim, state0, TARGET_LOCATION, kp_range, ki_range, kd_range)
    
    # -------------------------------------------------------------------------
    # METHOD 2: Fast gradient-based optimization
    # -------------------------------------------------------------------------
    # optimal_kp, optimal_ki, optimal_kd = optimize_pid_gradient(
    #     sim, state0, TARGET_LOCATION, initial_guess=(0.2, 0.05, 0.3)
    # )
    
    # -------------------------------------------------------------------------
    # METHOD 3: Thorough global optimization (recommended)
    # -------------------------------------------------------------------------
    # optimal_kp, optimal_ki, optimal_kd = optimize_pid_global(
    #     sim, state0, TARGET_LOCATION
    # )
    
    # -------------------------------------------------------------------------
    # METHOD 4: Compare multiple configurations
    # -------------------------------------------------------------------------
    # configs = [
    #     (0.2, 0.0, 0.0, "P only"),
    #     (0.5, 0.0, 0.3, "PD"),
    #     (0.3, 0.05, 0.4, "PID v1"),
    #     (optimal_kp, optimal_ki, optimal_kd, "Optimized")
    # ]
    # 
    # results_list = []
    # labels = []
    # for kp, ki, kd, label in configs:
    #     _, metrics = evaluate_controller(kp, ki, kd, sim, state0, TARGET_LOCATION)
    #     if metrics:
    #         results_list.append((metrics['times'], metrics['states'], metrics['distances']))
    #         labels.append(label)
    # 
    # plot_comparison(results_list, labels, TARGET_LOCATION)
    
    print("\n" + "="*70)
    print("TUNING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()