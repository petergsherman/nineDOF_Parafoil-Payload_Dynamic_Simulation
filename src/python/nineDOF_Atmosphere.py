from dataclasses import dataclass
import numpy as np
from scipy import signal

@dataclass
class staticAtmosphere:
    DEN: float = 1.22566     # Air density (kg/m^3)
    VXWIND: float = 0.0      # Wind velocity X (m/s)
    VYWIND: float = 0.0      # Wind velocity Y (m/s)
    VZWIND: float = 0.0      # Wind velocity Z (m/s)


class dynamicAtmosphere:
    """
    Dryden wind turbulence model implementation based on MIL-F-8785C and MIL-HDBK-1797.
    
    This model generates realistic atmospheric turbulence using forming filters driven
    by white noise, with turbulence intensities that vary with altitude and wind speed.
    
    The Dryden model uses rational transfer functions to shape white noise into
    correlated turbulence with appropriate power spectral densities.
    
    Usage:
        atm = dynamicAtmosphere(turbulence_intensity='moderate')
        ...
        altitude = -state[2]   # if z is down
        velocity = np.linalg.norm([vx, vy, vz])  # airspeed
        atm.update(t, altitude, velocity)
        # then use atm.DEN, atm.VXWIND, atm.VYWIND, atm.VZWIND
    """

    def __init__(
        self,
        turbulence_intensity: str = 'moderate',  # 'light', 'moderate', 'severe'
        mean_wind_speed: float = 0.0,
        mean_wind_direction: float = 0.0,  # degrees from north
        altitude_ref: float = 0.0,  # reference altitude for wind profile
        seed: int | None = None,
    ) -> None:
        """
        Initialize Dryden turbulence model.
        
        Args:
            turbulence_intensity: 'light', 'moderate', or 'severe'
            mean_wind_speed: Mean wind speed at reference altitude (m/s)
            mean_wind_direction: Mean wind direction in degrees (0=north, 90=east)
            altitude_ref: Reference altitude for mean wind (m)
            seed: Random seed for reproducibility
        """
        self.turbulence_intensity = turbulence_intensity
        self.mean_wind_speed = mean_wind_speed
        self.mean_wind_direction = np.deg2rad(mean_wind_direction)
        self.altitude_ref = altitude_ref
        
        # Public atmosphere state
        self.DEN: float = 1.22566
        self.VXWIND: float = 0.0
        self.VYWIND: float = 0.0
        self.VZWIND: float = 0.0
        
        # Turbulence scale factor (can reduce if causing issues)
        self.turbulence_scale = 1.0  # Set to 0.0 to disable turbulence, 0.5 for half strength
        
        # Random generator
        self.rng = np.random.default_rng(seed)
        
        # Turbulence state variables (for forming filters)
        self._u_gust_state = np.zeros(2)  # longitudinal gust states
        self._v_gust_state = np.zeros(2)  # lateral gust states
        self._w_gust_state = np.zeros(2)  # vertical gust states
        
        # Time tracking
        self._last_t: float | None = None

    def _get_turbulence_intensities(self, altitude: float) -> tuple[float, float, float]:
        """
        Get turbulence intensities (sigma_u, sigma_v, sigma_w) based on altitude
        and specified intensity level according to MIL-F-8785C.
        
        Returns: (sigma_u, sigma_v, sigma_w) in m/s
        """
        # Low altitude model (h < 1000 ft = 304.8 m)
        # Medium/high altitude model (h >= 1000 ft)
        
        h_ft = altitude * 3.28084  # convert to feet
        
        if self.turbulence_intensity == 'light':
            W20 = 15 * 0.514444  # 15 knots to m/s
        elif self.turbulence_intensity == 'moderate':
            W20 = 30 * 0.514444  # 30 knots to m/s
        elif self.turbulence_intensity == 'severe':
            W20 = 45 * 0.514444  # 45 knots to m/s
        else:
            W20 = 15 * 0.514444  # default to light
        
        if altitude < 304.8:  # Low altitude (< 1000 ft)
            # Use low altitude model
            sigma_w = 0.1 * W20
            sigma_u = sigma_w / (0.177 + 0.000823 * h_ft) ** 0.4
            sigma_v = sigma_u
        else:  # Medium/high altitude
            # Linear decrease with altitude
            h_m = altitude  # meters
            sigma_w = 0.1 * W20
            sigma_u = sigma_w
            sigma_v = sigma_w
            
            # Decrease intensity with altitude
            if h_m > 1000:
                scale = max(0.1, 1.0 - (h_m - 1000) / 10000)
                sigma_u *= scale
                sigma_v *= scale
                sigma_w *= scale
        
        return sigma_u, sigma_v, sigma_w

    def _get_scale_lengths(self, altitude: float) -> tuple[float, float, float]:
        """
        Get turbulence scale lengths (L_u, L_v, L_w) based on altitude
        according to MIL-F-8785C.
        
        Returns: (L_u, L_v, L_w) in meters
        """
        h_ft = altitude * 3.28084  # convert to feet
        h_m = altitude  # meters
        
        if altitude < 304.8:  # Low altitude (< 1000 ft)
            L_w = h_m  # scale with altitude
            L_u = L_w / (0.177 + 0.000823 * h_ft) ** 1.2
            L_v = L_u
        else:  # Medium/high altitude
            L_u = 533.4  # meters (1750 ft)
            L_v = L_u
            L_w = 533.4  # meters
        
        # Ensure minimum scale lengths
        L_u = max(L_u, 10.0)
        L_v = max(L_v, 10.0)
        L_w = max(L_w, 10.0)
        
        return L_u, L_v, L_w

    def _dryden_transfer_function_u(self, L: float, sigma: float, V: float, dt: float):
        """
        Discretize the Dryden transfer function for longitudinal turbulence.
        H_u(s) = sigma_u * sqrt(2*L_u/pi/V) / (1 + L_u*s/V)
        """
        if V < 1.0:
            V = 1.0  # Avoid division by zero
        
        # Continuous time parameters
        tau = L / V
        K = sigma * np.sqrt(2 * L / (np.pi * V))
        
        # Continuous transfer function: K / (tau*s + 1)
        # Discretize using Tustin (bilinear transform)
        num_c = [K]
        den_c = [tau, 1.0]
        
        sys_c = signal.TransferFunction(num_c, den_c)
        sys_d = sys_c.to_discrete(dt, method='bilinear')
        
        return sys_d.num.flatten(), sys_d.den.flatten()

    def _dryden_transfer_function_vw(self, L: float, sigma: float, V: float, dt: float):
        """
        Discretize the Dryden transfer function for lateral/vertical turbulence.
        H_v(s) = sigma_v * sqrt(L_v/pi/V) * (1 + sqrt(3)*L_v*s/V) / (1 + L_v*s/V)^2
        """
        if V < 1.0:
            V = 1.0
        
        tau = L / V
        K = sigma * np.sqrt(L / (np.pi * V))
        
        # Continuous: K * (1 + sqrt(3)*tau*s) / (tau*s + 1)^2
        num_c = [K * np.sqrt(3) * tau, K]
        den_c = [tau**2, 2*tau, 1.0]
        
        sys_c = signal.TransferFunction(num_c, den_c)
        sys_d = sys_c.to_discrete(dt, method='bilinear')
        
        return sys_d.num.flatten(), sys_d.den.flatten()

    def _filter_white_noise(self, white_noise: float, num: np.ndarray, den: np.ndarray, 
                           state: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Apply discrete filter to white noise input.
        Returns: (output, new_state)
        """
        # Ensure coefficient arrays have same length
        max_len = max(len(num), len(den))
        num_padded = np.pad(num, (0, max_len - len(num)))
        den_padded = np.pad(den, (0, max_len - len(den)))
        
        # Normalize by den[0]
        num_norm = num_padded / den_padded[0]
        den_norm = den_padded / den_padded[0]
        
        # Direct form II implementation
        n_states = len(den_norm) - 1
        if len(state) != n_states:
            state = np.zeros(n_states)
        
        # Compute output
        output = num_norm[0] * white_noise
        if n_states > 0:
            output += np.dot(num_norm[1:n_states+1], state[:n_states])
        
        # Update states
        new_state = state.copy()
        if n_states > 0:
            new_state[0] = white_noise - np.dot(den_norm[1:n_states+1], state[:n_states])
            if n_states > 1:
                new_state[1:] = state[:-1]
        
        return output, new_state

    def _get_mean_wind(self, altitude: float) -> tuple[float, float, float]:
        """
        Get mean wind components based on altitude using power law wind profile.
        """
        if altitude < 0:
            altitude = 0
        
        # Power law wind profile: V(h) = V_ref * (h/h_ref)^alpha
        # Typical alpha = 0.14 for neutral stability
        alpha = 0.14
        h_ref = max(self.altitude_ref, 10.0)  # minimum 10m reference
        
        if altitude < h_ref:
            wind_speed = self.mean_wind_speed * (altitude / h_ref) ** alpha
        else:
            wind_speed = self.mean_wind_speed
        
        # Convert to components (wind direction is "from" direction)
        # North is 0°, East is 90°
        # Wind FROM north means blowing TO south (negative Y in NED)
        wind_x = -wind_speed * np.sin(self.mean_wind_direction)  # East component
        wind_y = -wind_speed * np.cos(self.mean_wind_direction)  # North component
        wind_z = 0.0  # No mean vertical wind
        
        return wind_x, wind_y, wind_z

    def _density_from_alt(self, altitude: float) -> float:
        """
        Standard atmosphere density model.
        """
        rho0 = 1.22566  # kg/m^3 at sea level
        H = 8500.0      # scale height [m]
        alt_clamped = max(0.0, altitude)
        return rho0 * np.exp(-alt_clamped / H)

    def update(self, t: float, altitude: float, airspeed: float | None = None) -> tuple[float, float, float, float]:
        """
        Update the Dryden turbulence model.
        
        Args:
            t: Current time (s)
            altitude: Geometric altitude above ground (m), positive upwards
            airspeed: True airspeed magnitude (m/s). If None, uses a default of 15 m/s
        
        Returns:
            (density, wind_x, wind_y, wind_z)
        
        After calling, use self.DEN, self.VXWIND, self.VYWIND, self.VZWIND
        """
        # Use default airspeed if not provided
        if airspeed is None:
            airspeed = 15.0  # Default typical parafoil airspeed
        # Clamp altitude
        alt = float(max(0.0, altitude))
        
        # Time step
        if self._last_t is None:
            dt = 0.01  # Initial timestep
        else:
            dt = max(1e-3, min(t - self._last_t, 1.0))  # Clamp dt for stability
        self._last_t = t
        
        # Get turbulence parameters
        sigma_u, sigma_v, sigma_w = self._get_turbulence_intensities(alt)
        L_u, L_v, L_w = self._get_scale_lengths(alt)
        
        # Use airspeed for filter design (spatial frequency)
        V = max(airspeed, 1.0)  # Minimum 1 m/s to avoid singularities
        
        # Generate white noise inputs
        noise_u = self.rng.normal(0.0, 1.0)
        noise_v = self.rng.normal(0.0, 1.0)
        noise_w = self.rng.normal(0.0, 1.0)
        
        # Get discrete transfer functions
        num_u, den_u = self._dryden_transfer_function_u(L_u, sigma_u, V, dt)
        num_v, den_v = self._dryden_transfer_function_vw(L_v, sigma_v, V, dt)
        num_w, den_w = self._dryden_transfer_function_vw(L_w, sigma_w, V, dt)
        
        # Filter noise through transfer functions
        u_gust, self._u_gust_state = self._filter_white_noise(
            noise_u, num_u, den_u, self._u_gust_state
        )
        v_gust, self._v_gust_state = self._filter_white_noise(
            noise_v, num_v, den_v, self._v_gust_state
        )
        w_gust, self._w_gust_state = self._filter_white_noise(
            noise_w, num_w, den_w, self._w_gust_state
        )
        
        # Apply turbulence scale factor
        u_gust *= self.turbulence_scale
        v_gust *= self.turbulence_scale
        w_gust *= self.turbulence_scale
        
        # Clamp turbulence to reasonable limits (prevent numerical issues)
        max_gust = 3.0 * max(sigma_u, sigma_v, sigma_w)
        u_gust = np.clip(u_gust, -max_gust, max_gust)
        v_gust = np.clip(v_gust, -max_gust, max_gust)
        w_gust = np.clip(w_gust, -max_gust, max_gust)
        
        # Get mean wind
        mean_x, mean_y, mean_z = self._get_mean_wind(alt)
        
        # Total wind = mean + turbulence
        self.VXWIND = float(mean_x + u_gust)
        self.VYWIND = float(mean_y + v_gust)
        self.VZWIND = float(mean_z + w_gust)
        self.DEN = float(self._density_from_alt(alt))
        
        return self.DEN, self.VXWIND, self.VYWIND, self.VZWIND