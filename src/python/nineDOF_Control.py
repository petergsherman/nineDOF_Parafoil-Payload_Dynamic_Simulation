#nineDOF_Control.py
import numpy as np
from scipy.linalg import solve_continuous_are

# ============================================================================
# UTILITY
# ============================================================================

def make_control_function(controller):
    """
    Wraps a controller object's computeControl() method into the callable
    signature expected by plant.run_simulation():  f(t, state) -> (dL, dR, inc)
    """
    def control_wrapper(t, state):
        return controller.computeControl(state)
    return control_wrapper


# ============================================================================
# LINEARIZATION HELPER
# ============================================================================

def linearize_plant(plant_obj, state_trim, deltaL_trim, deltaR_trim, incidence_trim,
                    eps_state=1e-5, eps_ctrl=1e-5):
    """
    Numerically linearize plant.computeDerivatives() around a trim point.

    Returns
    -------
    A : (18, 18)  state Jacobian   (dx_dot / dx)
    B : (18,  3)  control Jacobian (dx_dot / du),  u = [deltaL, deltaR, incidence]
    """
    n = len(state_trim)
    m = 3   # deltaL, deltaR, incidence

    f0 = plant_obj.computeDerivatives(state_trim, deltaL_trim, deltaR_trim, incidence_trim)

    # State Jacobian  A
    A = np.zeros((n, n))
    for i in range(n):
        s_hi = state_trim.copy();  s_hi[i] += eps_state
        s_lo = state_trim.copy();  s_lo[i] -= eps_state
        fhi = plant_obj.computeDerivatives(s_hi, deltaL_trim, deltaR_trim, incidence_trim)
        flo = plant_obj.computeDerivatives(s_lo, deltaL_trim, deltaR_trim, incidence_trim)
        A[:, i] = (fhi - flo) / (2.0 * eps_state)

    # Control Jacobian  B
    B = np.zeros((n, m))
    u_trim = np.array([deltaL_trim, deltaR_trim, incidence_trim])
    for j in range(m):
        u_hi = u_trim.copy();  u_hi[j] += eps_ctrl
        u_lo = u_trim.copy();  u_lo[j] -= eps_ctrl
        fhi = plant_obj.computeDerivatives(state_trim, u_hi[0], u_hi[1], u_hi[2])
        flo = plant_obj.computeDerivatives(state_trim, u_lo[0], u_lo[1], u_lo[2])
        B[:, j] = (fhi - flo) / (2.0 * eps_ctrl)

    return A, B


def solve_lqr(A, B, Q, R):
    """
    Solve the continuous-time infinite-horizon LQR problem.

    min  integral( x'Qx + u'Ru ) dt

    Returns K such that  u = -K x  is optimal.
    Raises ValueError if the Riccati equation fails.
    """
    try:
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P)   # K = R^{-1} B^T P
        return K
    except Exception as exc:
        raise ValueError(f"CARE solver failed: {exc}")


# ============================================================================
# TRIM FINDER
# ============================================================================

def find_trim(plant_obj, airspeed=10.0, max_iter=200, tol=1e-6):
    """
    Find a steady gliding trim state for the parafoil.

    The trim search is performed in the longitudinal plane:
    unknowns are  [thetaP, thetaC, incidence]  that minimise the norm of the
    longitudinal accelerations (u_dot, w_dot, qP_dot, qC_dot).

    Returns
    -------
    state_trim  : (18,) trim state  (psiP = 0, all lateral states = 0)
    deltaL_trim : trim deltaL (symmetric: deltaL = deltaR = 0)
    deltaR_trim : trim deltaR
    incidence_trim : trim incidence angle
    """
    from scipy.optimize import fsolve

    params = plant_obj.parameters

    def residuals(x):
        thetaP, incidence = x
        state = np.zeros(18)
        state[3]  = 0.0       # phiP
        state[4]  = thetaP    # thetaP
        state[5]  = 0.0       # psiP
        state[6]  = 0.0       # phiC
        state[7]  = thetaP    # thetaC ≈ thetaP at trim (cradle follows parafoil)
        state[8]  = 0.0       # psiC
        state[9]  = airspeed  # uG
        state[10] = 0.0       # vG
        state[11] = 0.0       # wG  — will be non-zero at trim generally

        deltaL = 0.0
        deltaR = 0.0

        sdot = plant_obj.computeDerivatives(state, deltaL, deltaR, incidence)

        # We want: uG_dot ≈ 0, wG_dot ≈ 0  (steady glide, no acceleration)
        return [sdot[9], sdot[11]]

    # Initial guess: slight nose-down pitch, nominal incidence
    x0 = np.array([-0.05, params.NOM_INCIDENCE])
    sol, info, ier, msg = fsolve(residuals, x0, full_output=True)

    if ier != 1:
        print(f"[LQR] Trim solver warning (ier={ier}): {msg} — using initial guess.")
        sol = x0

    thetaP_trim, incidence_trim = sol
    state_trim = np.zeros(18)
    state_trim[4]  = thetaP_trim
    state_trim[7]  = thetaP_trim
    state_trim[9]  = airspeed
    # Steady-state w from the residuals (not forced to zero)
    state_trim[11] = 0.0

    print(f"[LQR] Trim: thetaP={np.degrees(thetaP_trim):.2f} deg, "
          f"incidence={np.degrees(incidence_trim):.2f} deg")

    return state_trim, 0.0, 0.0, incidence_trim


# ============================================================================
# STATE / CONTROL INDEX CONVENTIONS
# ============================================================================
#
# State vector (18 elements)
#   0-2   : xG, yG, zG          inertial position
#   3-5   : phiP, thetaP, psiP  parafoil Euler angles
#   6-8   : phiC, thetaC, psiC  cradle  Euler angles
#   9-11  : uG, vG, wG          gimbal velocity in parafoil frame
#   12-14 : pP, qP, rP          parafoil angular rates
#   15-17 : pC, qC, rC          cradle  angular rates
#
# Control vector u = [deltaL, deltaR, incidence]
#   differential brake  dA = deltaR - deltaL   (positive → right turn)
#   symmetric brake     dS = 0.5(deltaL+deltaR)
#
# For lateral-directional LQR we select the state subset:
#   phiP(3), psiP(5), vG(10), pP(12), rP(14), pC(15), rC(17)
# and the single control channel:
#   u_lqr = dA = deltaR - deltaL
#
# ============================================================================

# Indices of the two heading states
LAT_IDX = np.array([5, 14])
#                   psiP rP

LAT_NAMES = ["psiP", "rP"]
N_LAT = len(LAT_IDX)  # 2


# ============================================================================
# FULL LINEARIZED LQR HEADING CONTROLLER
# ============================================================================

class LQRHeadingController:
    """
    Linearized LQR controller for parafoil heading guidance.

    Architecture
    ------------
    Outer loop (guidance):
        Computes a desired heading psi_d from the current position and target.

    Inner loop (LQR):
        Regulates only parafoil heading error and heading rate error:
            x = [psiP_error, rP_error]
        using a 2-state gain K computed from the linearized plant at trim.

        Control:  dA = -K @ [psi_error, rP]   (differential brake)
            deltaL = max(-dA, 0)
            deltaR = max( dA, 0)

    Parameters
    ----------
    targetLandingLocation : (x, y)  target in inertial XY plane
    plant_obj             : nineDOF_Plant.plant instance (needed for linearization)
                            If None, falls back to an analytical double-integrator gain.
    Q : (2, 2) state cost matrix   — penalises [psiP_error, rP_error]
    R : (1, 1) control cost matrix — penalises control effort
    max_control : float   maximum brake deflection (m)
    airspeed    : float   trim airspeed used for linearization (m/s)
    """

    def __init__(self, targetLandingLocation,
                 plant_obj=None,
                 Q=None, R=None,
                 max_control=0.94,
                 airspeed=10.0):

        self.target       = np.array(targetLandingLocation, dtype=float)
        self.max_control  = max_control
        self.airspeed     = airspeed
        self.plant_obj    = plant_obj

        # ---- Cost matrices -----------------------------------------------
        # Q penalises:  psiP_error,  rP_error
        if Q is None:
            Q = np.diag([20.0, 2.0])
        if R is None:
            R = np.array([[2.0]])

        self.Q = Q
        self.R = R

        # ---- Compute LQR gain --------------------------------------------
        self.K = self._compute_gain()

        print(f"[LQR] Gain K = {self.K}")

    # ------------------------------------------------------------------
    def _compute_gain(self):
        """
        Build the reduced lateral-directional (A_lat, B_lat) matrices and
        solve the LQR Riccati equation.
        """
        if self.plant_obj is not None:
            return self._gain_from_linearization()
        else:
            return self._gain_analytical_fallback()

    # ------------------------------------------------------------------
    def _gain_from_linearization(self):
        """Linearize plant numerically and extract lateral subsystem."""
        print("[LQR] Linearizing plant at trim …")
        state_trim, dL_trim, dR_trim, inc_trim = find_trim(
            self.plant_obj, airspeed=self.airspeed)

        A_full, B_full = linearize_plant(
            self.plant_obj, state_trim, dL_trim, dR_trim, inc_trim)

        # Extract 2-state heading subsystem: [psiP(5), rP(14)]
        # dA = dR - dL  →  B_lat = B_full[LAT_IDX, 1] - B_full[LAT_IDX, 0]
        A_lat = A_full[np.ix_(LAT_IDX, LAT_IDX)]
        B_lat = (B_full[LAT_IDX, 1] - B_full[LAT_IDX, 0]).reshape(-1, 1)

        print(f"[LQR] A_lat =\n{A_lat}")
        print(f"[LQR] B_lat = {B_lat.T}")

        # Check controllability
        C_mat = self._controllability_matrix(A_lat, B_lat)
        rank = np.linalg.matrix_rank(C_mat)
        if rank < N_LAT:
            print(f"[LQR] Warning: heading subsystem controllability rank {rank}/{N_LAT}. "
                  "Falling back to analytical gains.")
            return self._gain_analytical_fallback()

        try:
            K = solve_lqr(A_lat, B_lat, self.Q, self.R)
            return K
        except ValueError as e:
            print(f"[LQR] {e} — falling back to analytical gains.")
            return self._gain_analytical_fallback()

    # ------------------------------------------------------------------
    @staticmethod
    def _controllability_matrix(A, B):
        n = A.shape[0]
        cols = [B]
        for _ in range(n - 1):
            cols.append(A @ cols[-1])
        return np.hstack(cols)

    # ------------------------------------------------------------------
    def _gain_analytical_fallback(self):
        """
        Analytical LQR for the 2-state heading double-integrator:
            x = [psiP_error, rP]
            psiP_dot = rP
            rP_dot   = b * dA   (yaw moment from differential brake)
        """
        print("[LQR] Using analytical fallback gain.")
        A_lat = np.array([[0.0, 1.0],
                          [0.0, 0.0]])
        b_yaw = 0.2   # rad/s^2 per unit differential brake
        B_lat = np.array([[0.0], [b_yaw]])

        try:
            K = solve_lqr(A_lat, B_lat, self.Q, self.R)
            return K
        except ValueError as e:
            print(f"[LQR] Analytical fallback also failed: {e}. Using hand-tuned PD gains.")
            # K shape (1, 2): [psiP_error, rP]
            return np.array([[8.0, 2.0]])

    # ------------------------------------------------------------------
    def computeControl(self, state):
        """
        Compute (deltaL, deltaR, incidence) given the full 18-element state.

        Outer guidance loop: compute desired heading psi_d to the target.
        Inner LQR loop: regulate lateral state error with u = -K @ delta_x.
        """
        # ---- Position and heading ----------------------------------------
        pos_xy         = state[0:2]
        psiP           = state[5]

        to_target = self.target - pos_xy
        dist      = np.linalg.norm(to_target)

        # ---- Desired heading ----------------------------------------------
        if dist < 1.0:
            # Already at target — hold wings level
            return (0.0, 0.0, 0.0)

        psi_d = np.arctan2(to_target[1], to_target[0])

        # ---- Heading error (wrapped to [-pi, pi]) -------------------------
        delta_psi = np.arctan2(np.sin(psiP - psi_d), np.cos(psiP - psi_d))

        # ---- Heading rate -------------------------------------------------
        rP = state[14]   # parafoil yaw rate

        # ---- LQR control law: u = -K @ [psi_error, rP] -------------------
        x_err = np.array([delta_psi, rP])
        dA = float(np.squeeze(-self.K @ x_err))  # differential brake  dA = dR - dL

        # Saturate
        dA = np.clip(dA, -self.max_control, self.max_control)

        # Map differential brake to individual brakes
        if dA >= 0.0:
            deltaR = dA
            deltaL = 0.0
        else:
            deltaL = -dA
            deltaR = 0.0

        incidence = 0.0
        return (deltaL, deltaR, incidence)

    # ------------------------------------------------------------------
    def set_cost_matrices(self, Q, R):
        """Recompute the LQR gain with new cost matrices."""
        self.Q = Q
        self.R = R
        self.K = self._compute_gain()
        print(f"[LQR] Gain recomputed: K = {self.K}")

    def get_gain(self):
        return self.K.copy()


# ============================================================================
# SIMPLE / TEST CONTROLLERS  (retained for compatibility)
# ============================================================================

class testController:
    def __init__(self) -> None:
        self.deltaL = 0.0
        self.deltaR = 0.0
        self.incidence = 0.0

    def computeControl(self, state):
        altitude = -state[2]
        if altitude < 250:
            return (0.9, 0.0, 0.0)
        return (0.0, 0.0, 0.0)


class simpleHeadingController:
    """Bang-bang heading controller (original)."""
    def __init__(self, targetLandingLocation) -> None:
        self.target = np.array(targetLandingLocation, dtype=float)

    def computeControl(self, state):
        heading_I      = np.array([np.cos(state[5]), np.sin(state[5])])
        to_target      = self.target - state[0:2]
        dist           = np.linalg.norm(to_target)
        if dist < 1.0:
            return (0.0, 0.0, 0.0)
        headingTarget_I = to_target / dist
        cross = np.cross(heading_I, headingTarget_I)
        if cross > 0:
            return (0.0, float(np.abs(cross)), 0.0)
        elif cross < 0:
            return (float(np.abs(cross)), 0.0, 0.0)
        return (0.0, 0.0, 0.0)


class PIDHeadingController:
    """
    PID heading controller (original, preserved for comparison).
    """
    def __init__(self, targetLandingLocation,
                 kp=0.5, ki=0.01, kd=0.1,
                 max_control=0.94, dt=0.01):
        self.target        = np.array(targetLandingLocation, dtype=float)
        self.kp            = kp
        self.ki            = ki
        self.kd            = kd
        self.max_control   = max_control
        self.dt            = dt
        self.integral_error = 0.0
        self.prev_error    = 0.0
        self.prev_time     = None
        self.first_call    = True

    def computeControl(self, state, current_time=None):
        to_target      = self.target - state[0:2]
        dist           = np.linalg.norm(to_target)
        if dist < 5.0:
            return (0.0, 0.0, 0.0)

        desired_heading = np.arctan2(to_target[1], to_target[0])
        heading_error   = np.arctan2(
            np.sin(desired_heading - state[5]),
            np.cos(desired_heading - state[5]))

        dt = self.dt
        if current_time is not None and self.prev_time is not None:
            dt = max(current_time - self.prev_time, 1e-6)

        p_term = self.kp * heading_error

        i_term = 0.0
        if not self.first_call:
            self.integral_error = np.clip(
                self.integral_error + heading_error * dt, -10.0, 10.0)
            i_term = self.ki * self.integral_error

        d_term = 0.0
        if not self.first_call and dt > 0:
            d_term = self.kd * (heading_error - self.prev_error) / dt

        self.prev_error = heading_error
        self.prev_time  = current_time
        self.first_call = False

        u = np.clip(p_term + i_term + d_term,
                    -self.max_control, self.max_control)

        if u > 0:
            return (u,  0.0, 0.0)
        else:
            return (0.0, -u, 0.0)

    def reset(self):
        self.integral_error = 0.0
        self.prev_error     = 0.0
        self.prev_time      = None
        self.first_call     = True