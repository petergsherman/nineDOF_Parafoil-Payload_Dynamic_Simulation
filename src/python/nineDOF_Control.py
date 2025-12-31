#nineDOF_Control.py
import numpy as np
from scipy.linalg import solve_continuous_are

class testController:
    def __init__(self) -> None:
        self.deltaL = 0.0
        self.deltaR = 0.0
        self.incidence = 0.0

    def computeControl(self, state):
        altitude = -state[2]  # Convert to positive altitude
        print (state[0], state[1], state[2], sep=", ")
        if altitude > 250:
            return (0.0, 0.0, 0.0)
        elif altitude < 250:
            return (0.9, 0.0, 0.0)
        else:
            return (0.0, 0.0, 0.0)

class simpleHeadingController:
    def __init__(self, targetLandingLocation) -> None:
        self.deltaL = 0.0
        self.deltaR = 0.0
        self.incidence = 0.0
        self.target = targetLandingLocation
    
    def computeControl(self, state):
        #Calculating Parafoil Heading Unit Vector 
        heading_I = np.array([np.cos(state[5]), np.sin(state[5])])

        #Calculating Inertial Heading Unit Vector
        headingTarget_I = np.array([self.target[0] - state[0], self.target[1] - state[1]]) / np.linalg.norm(np.array([self.target[0] - state[0], self.target[1] - state[1]]))
        
        #Computing Cross Product of Heading and Target Heading
        cross = np.cross(heading_I[0:2], headingTarget_I[0:2])
        #print(heading_I[0], heading_I[1], headingTarget_I[0], headingTarget_I[1], sep=', ')

        if cross > 0:
            return (0, np.abs(cross), 0) #Right Break Control
        elif cross < 0:
            return (np.abs(cross), 0, 0) #Left Break Control
        
class PIDHeadingController:
    """
    PID controller for parafoil heading control to guide system to target landing point.
    Uses heading error to compute brake control inputs (deltaL, deltaR).
    """
    def __init__(self, targetLandingLocation, kp=0.5, ki=0.01, kd=0.2, max_control=0.94):
        """
        Args:
            targetLandingLocation: Tuple (x, y) of target landing coordinates
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            max_control: Maximum control deflection (0 to 0.94)
        """
        self.target = np.array(targetLandingLocation)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_control = max_control
        
        # PID state variables
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        
    def computeControl(self, state):
        """
        Compute control based on current state.
        
        Args:
            state: State vector [x, y, z, phi_p, theta_p, psi_p, ...]
                   where psi_p (state[5]) is the parafoil heading angle
        
        Returns:
            Tuple (deltaL, deltaR, incidence) - control inputs
        """
        # Extract current position and heading
        current_pos = np.array([state[0], state[1]])
        current_heading = state[5]  # psi_p - parafoil heading angle
        
        # Calculate vector to target
        to_target = self.target - current_pos
        distance_to_target = np.linalg.norm(to_target)
        
        # If very close to target, no control needed
        if distance_to_target < 10.0:
            return (0.0, 0.9, 0.0)
        
        # Desired heading angle to target
        desired_heading = np.arctan2(to_target[1], to_target[0])
        
        # Compute heading error (wrapped to [-pi, pi])
        heading_error = desired_heading - current_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # PID calculations
        # Proportional term
        p_term = self.kp * heading_error
        
        # Integral term (with anti-windup)
        self.integral_error += heading_error
        # Anti-windup: limit integral to prevent excessive buildup
        max_integral = 10.0
        self.integral_error = np.clip(self.integral_error, -max_integral, max_integral)
        i_term = self.ki * self.integral_error
        
        # Derivative term
        error_rate = heading_error - self.prev_error
        d_term = self.kd * error_rate
        self.prev_error = heading_error
        
        # Total control output
        control_output = p_term + i_term + d_term
        
        # Map control output to brake deflections
        # Positive error -> turn left (apply left brake)
        # Negative error -> turn right (apply right brake)
        if control_output > 0:
            deltaL = np.clip(control_output, 0.0, self.max_control)
            deltaR = 0.0
        else:
            deltaL = 0.0
            deltaR = np.clip(-control_output, 0.0, self.max_control)
        
        # Incidence control (could be used for altitude control, currently 0)
        incidence = 0.0
        
        return (deltaL, deltaR, incidence)
    
    def reset(self):
        """Reset the controller state (useful for multiple runs)"""
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_time = None


class LQRHeadingController:
    """
    Linear Quadratic Regulator (LQR) controller for parafoil guidance.
    Controls heading and heading rate to guide system to target landing point.
    """
    def __init__(self, targetLandingLocation, Q=None, R=None, max_control=0.94):
        """
        Args:
            targetLandingLocation: Tuple (x, y) of target landing coordinates
            Q: State cost matrix (2x2) - penalizes [heading_error, heading_rate_error]
            R: Control cost matrix (1x1) - penalizes control effort
            max_control: Maximum control deflection (0 to 0.94)
        """
        self.target = np.array(targetLandingLocation)
        self.max_control = max_control
        
        # Default cost matrices if not provided
        if Q is None:
            # Penalize heading error heavily, heading rate moderately
            Q = np.array([[10.0, 0.0],
                         [0.0, 1.0]])
        if R is None:
            # Moderate control effort penalty
            R = np.array([[1.0]])
        
        self.Q = Q
        self.R = R
        
        # Compute LQR gain
        self.K = self._compute_lqr_gain()
        
    def _compute_lqr_gain(self):
        """
        Compute LQR gain matrix for linearized heading dynamics.
        
        Simplified model:
        State: x = [heading_error, heading_rate]
        Control: u = [differential_brake]
        
        Dynamics (linearized):
        dx/dt = A*x + B*u
        
        A = [0,  1]     (heading_rate affects heading)
            [0,  0]     (simplified: no natural damping)
        
        B = [0]         (control doesn't directly affect heading)
            [k]         (control affects heading rate, k is effectiveness)
        """
        # Linearized system matrices
        # State: [heading_error, heading_rate_error]
        A = np.array([[0.0, 1.0],
                     [0.0, 0.0]])
        
        # Control effectiveness (tune this based on your system)
        control_effectiveness = 0.5  # rad/s per unit control
        B = np.array([[0.0],
                     [control_effectiveness]])
        
        # Solve continuous-time algebraic Riccati equation
        try:
            P = solve_continuous_are(A, B, self.Q, self.R)
            # Compute optimal gain: K = R^-1 * B^T * P
            K = np.linalg.inv(self.R) @ B.T @ P
            return K
        except Exception as e:
            print(f"Warning: LQR gain computation failed: {e}")
            print("Using fallback proportional-derivative gains")
            # Fallback to simple PD-like gains
            return np.array([[5.0, 2.0]])
    
    def computeControl(self, state):
        """
        Compute control based on current state using LQR feedback.
        
        Args:
            state: State vector [x, y, z, phi_p, theta_p, psi_p, phi_c, theta_c, psi_c,
                                 u_p, v_p, w_p, u_c, v_c, w_c, p, q, r]
                   where:
                   - psi_p (state[5]) is the parafoil heading angle
                   - r (state[17]) is the yaw rate
        
        Returns:
            Tuple (deltaL, deltaR, incidence) - control inputs
        """
        # Extract current position and heading
        current_pos = np.array([state[0], state[1]])
        current_heading = state[5]  # psi_p - parafoil heading angle
        
        # Extract heading rate (yaw rate)
        if len(state) > 17:
            heading_rate = state[17]  # r - yaw rate
        else:
            heading_rate = 0.0
        
        # Calculate vector to target
        to_target = self.target - current_pos
        distance_to_target = np.linalg.norm(to_target)
        
        # If very close to target, no control needed
        if distance_to_target < 10.0:
            return (0.0, 0.0, 0.0)
        
        # Desired heading angle to target
        desired_heading = np.arctan2(to_target[1], to_target[0])
        
        # Compute heading error (wrapped to [-pi, pi])
        heading_error = desired_heading - current_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Desired heading rate (typically zero for steady tracking)
        desired_heading_rate = 0.0
        heading_rate_error = desired_heading_rate - heading_rate
        
        # State error vector
        x_error = np.array([heading_error, heading_rate_error])
        
        # LQR control law: u = -K * x_error
        control_output = -self.K @ x_error
        control_output = control_output[0]  # Extract scalar
        
        # Saturate control to maximum deflection
        control_output = np.clip(control_output, -self.max_control, self.max_control)
        
        # Map control output to brake deflections
        # Positive control -> turn left (apply left brake)
        # Negative control -> turn right (apply right brake)
        if control_output > 0:
            deltaL = control_output
            deltaR = 0.0
        else:
            deltaL = 0.0
            deltaR = -control_output
        
        # Incidence control (for altitude control, currently unused)
        incidence = 0.0
        
        return (deltaL, deltaR, incidence)
    
    def set_cost_matrices(self, Q, R):
        """
        Update cost matrices and recompute LQR gain.
        
        Args:
            Q: State cost matrix (2x2)
            R: Control cost matrix (1x1)
        """
        self.Q = Q
        self.R = R
        self.K = self._compute_lqr_gain()
        print(f"LQR gain updated: K = {self.K}")
    
    def get_gain(self):
        """Return the computed LQR gain matrix"""
        return self.K

def make_control_function(controller):
    """
    Wraps a controller object to work with plant.run_simulation()
    
    Usage:
        controller = PDController(target_altitude=-50.0)
        control_func = make_control_function(controller)
        times, states = sim.run_simulation(state0, t_final, dt, control_func)
    """
    def control_wrapper(t, state):
        return controller.computeControl(state)
    return control_wrapper