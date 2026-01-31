"""
Simple PID Tuning Script - Find optimal gains automatically

Usage:
    python tune_pid.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
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

# Simulation parameters
T_FINAL = 25.0
DT = 0.01
TARGET = (1000.0, 1000.0)

# ============================================================================
# OPTIMIZER
# ============================================================================

def evaluate_pid(params, sim, state0, target, dt=DT, t_final=T_FINAL):
    """Run simulation and return cost (lower is better)"""
    
    kp, ki, kd = params
    
    # Create controller
    controller = PIDHeadingController(target, kp=kp, ki=ki, kd=kd, dt=dt)
    control_func = make_control_function(controller)
    
    try:
        # Run simulation
        times, states = sim.run_simulation(state0, t_final, dt, control_func)
        
        # Calculate heading errors throughout flight
        heading_errors = []
        for state in states:
            current_pos = np.array([state[0], state[1]])
            current_heading = state[5]  # psi_p
            
            # Desired heading to target
            to_target = target - current_pos
            desired_heading = np.arctan2(to_target[1], to_target[0])
            
            # Heading error (wrapped to [-pi, pi])
            heading_error = desired_heading - current_heading
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            heading_errors.append(abs(heading_error))
        
        heading_errors = np.array(heading_errors)
        
        # Calculate metrics
        avg_heading_error = np.mean(heading_errors)
        max_heading_error = np.max(heading_errors)
        rms_heading_error = np.sqrt(np.mean(heading_errors**2))
        
        # Final heading error (important for landing alignment)
        final_heading_error = heading_errors[-1]
        
        # Combined cost: heavily weight heading tracking performance
        cost = (
            10.0 * rms_heading_error +           # Penalize RMS heading error heavily
            5.0 * avg_heading_error +             # Penalize average error
            3.0 * final_heading_error +           # Penalize final alignment error
            1.0 * max_heading_error               # Penalize worst-case error
        )
        
        return cost
        
    except:
        return 1e10  # Return huge cost if simulation fails


def find_optimal_pid(sim, state0, target):
    """
    Find optimal PID gains using global optimization.
    Returns: (optimal_kp, optimal_ki, optimal_kd)
    """
    
    print("="*60)
    print("Finding Optimal PID Gains...")
    print("="*60)
    
    def cost_function(params):
        kp, ki, kd = params
        cost = evaluate_pid(params, sim, state0, target)
        print(f"Testing: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f} -> Cost={cost:.2f}")
        return cost
    
    # Search bounds for each gain
    bounds = [
        (0.05, 2.0),   # Kp: 0.05 to 2.0
        (0.0, 0.3),    # Ki: 0 to 0.3 (keep small to avoid windup)
        (0.0, 2.0)     # Kd: 0 to 2.0
    ]
    
    # Run optimization
    result = differential_evolution(
        cost_function,
        bounds=bounds,
        maxiter=20,      # Number of generations
        popsize=8,       # Population size per generation
        seed=42,         # For reproducibility
        disp=True
    )
    
    optimal_kp, optimal_ki, optimal_kd = result.x
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"Optimal Kp: {optimal_kp:.4f}")
    print(f"Optimal Ki: {optimal_ki:.4f}")
    print(f"Optimal Kd: {optimal_kd:.4f}")
    print(f"Final Cost: {result.fun:.2f}")
    print("="*60)
    
    return optimal_kp, optimal_ki, optimal_kd


def plot_result(sim, state0, target, kp, ki, kd, dt=DT, t_final=T_FINAL):
    """Plot the trajectory with optimized gains"""
    
    controller = PIDHeadingController(target, kp=kp, ki=ki, kd=kd, dt=dt)
    control_func = make_control_function(controller)
    
    times, states = sim.run_simulation(state0, t_final, dt, control_func)
    
    # Calculate distances
    distances = np.linalg.norm(states[:, 0:2] - np.array(target), axis=1)
    
    # Calculate heading errors
    heading_errors = []
    for state in states:
        current_pos = np.array([state[0], state[1]])
        current_heading = state[5]
        to_target = target - current_pos
        desired_heading = np.arctan2(to_target[1], to_target[0])
        heading_error = desired_heading - current_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        heading_errors.append(np.rad2deg(heading_error))
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trajectory
    ax = axes[0, 0]
    ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='Flight Path')
    ax.plot(state0[0], state0[1], 'go', markersize=10, label='Start')
    ax.plot(target[0], target[1], 'r*', markersize=20, label='Target')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Trajectory\nKp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Distance over time
    ax = axes[0, 1]
    ax.plot(times, distances, 'b-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance to Target (m)')
    ax.set_title(f'Distance to Target\nFinal: {distances[-1]:.2f}m')
    ax.grid(True, alpha=0.3)
    
    # Heading error over time
    ax = axes[1, 0]
    ax.plot(times, heading_errors, 'r-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading Error (degrees)')
    ax.set_title(f'Heading Error\nRMS: {np.sqrt(np.mean(np.array(heading_errors)**2)):.2f}°')
    ax.grid(True, alpha=0.3)
    
    # Heading error statistics
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    Performance Metrics:
    
    Heading Error:
      • RMS Error: {np.sqrt(np.mean(np.array(heading_errors)**2)):.2f}°
      • Mean Error: {np.mean(np.abs(heading_errors)):.2f}°
      • Max Error: {np.max(np.abs(heading_errors)):.2f}°
      • Final Error: {abs(heading_errors[-1]):.2f}°
    
    Distance:
      • Final Distance: {distances[-1]:.2f} m
      • Mean Distance: {np.mean(distances):.2f} m
    
    PID Gains:
      • Kp = {kp:.4f}
      • Ki = {ki:.4f}
      • Kd = {kd:.4f}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('pid_result.png', dpi=150)
    plt.show()
    
    print(f"\nPlot saved to 'pid_result.png'")
    print(f"RMS heading error: {np.sqrt(np.mean(np.array(heading_errors)**2)):.2f}°")
    print(f"Final distance to target: {distances[-1]:.2f}m")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the optimizer"""
    
    # TODO: Uncomment and configure these lines
    # from your_plant_module import YourPlant
    # sim = YourPlant()
    # state0 = np.array([100, 100, -300, 0, 0, 0, ...])  # Your initial state
    
    print("\n*** PID Optimizer ***\n")
    print("This will find optimal PID gains for your parafoil controller.")
    print("Make sure to configure sim, state0, and TARGET above.\n")
    
    # Find optimal gains
    optimal_kp, optimal_ki, optimal_kd = find_optimal_pid(sim, state0, TARGET)
    
    # Plot the result
    plot_result(sim, state0, TARGET, optimal_kp, optimal_ki, optimal_kd)
    
    print("\nDone! Use these gains in your PIDHeadingController.")

if __name__ == "__main__":
    main()