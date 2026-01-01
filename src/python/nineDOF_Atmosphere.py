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
    Dryden wind turbulence model implementation based on MIL-F-8785C and MIL-HDBK-1797,
    with realistic layered wind structure that varies with altitude.
    
    This model generates:
    - Layered mean wind profile with varying speed and direction (like real atmosphere)
    - Realistic atmospheric turbulence using forming filters driven by white noise
    - Turbulence intensities that vary with altitude and wind speed
    
    The wind layers represent realistic atmospheric structure where wind speed and
    direction change with altitude due to different air masses, terrain effects,
    and atmospheric stability.
    
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
        altitude_max: float = 10000.0,  # Maximum altitude for wind profile (m)
        n_layers: int = 20,  # Number of distinct wind layers
        seed: int | None = None,
    ) -> None:
        """
        Initialize Dryden turbulence model with layered winds.
        
        Args:
            turbulence_intensity: 'light', 'moderate', or 'severe'
            altitude_max: Maximum altitude for wind layer generation (m)
            n_layers: Number of wind layers to generate
            seed: Random seed for reproducibility
        """
        self.turbulence_intensity = turbulence_intensity
        self.altitude_max = altitude_max
        self.n_layers = n_layers
        
        # Random generator (initialize before generating random surface conditions)
        self.rng = np.random.default_rng(seed)
        
        # Generate random surface wind conditions
        # Surface wind speed: typically 0-15 m/s (0-30 knots) for general conditions
        # Can be higher in storms, but we'll use moderate range
        self.surface_wind_speed = self.rng.uniform(2.0, 12.0)
        
        # Surface wind direction: completely random (0-360 degrees)
        self.surface_wind_direction = self.rng.uniform(0.0, 360.0)
        
        # Public atmosphere state
        self.DEN: float = 1.22566
        self.VXWIND: float = 0.0
        self.VYWIND: float = 0.0
        self.VZWIND: float = 0.0
        
        # Turbulence scale factor (can reduce if causing issues)
        self.turbulence_scale = 1.0  # Set to 0.0 to disable turbulence, 0.5 for half strength
        
        # Generate layered wind profile (must come after random surface conditions)
        self._generate_wind_layers()
        
        # Turbulence state variables (for forming filters)
        self._u_gust_state = np.zeros(2)  # longitudinal gust states
        self._v_gust_state = np.zeros(2)  # lateral gust states
        self._w_gust_state = np.zeros(2)  # vertical gust states
        
        # Time tracking
        self._last_t: float | None = None

    def _generate_wind_layers(self) -> None:
        """
        Generate realistic wind layers with varying speed and direction.
        
        Wind structure simulates:
        - Surface layer (0-100m): Strong shear, direction changes due to friction
        - Boundary layer (100-1000m): Moderate shear, backing/veering
        - Free atmosphere (1000m+): More uniform, can have jet streams
        """
        # Altitude grid for layers
        self.layer_altitudes = np.linspace(0, self.altitude_max, self.n_layers)
        
        # Initialize arrays
        self.layer_speeds = np.zeros(self.n_layers)
        self.layer_directions = np.zeros(self.n_layers)  # radians
        
        # Surface conditions
        self.layer_speeds[0] = self.surface_wind_speed
        self.layer_directions[0] = np.deg2rad(self.surface_wind_direction)
        
        # Generate wind profile with realistic transitions
        for i in range(1, self.n_layers):
            alt = self.layer_altitudes[i]
            alt_prev = self.layer_altitudes[i-1]
            delta_alt = alt - alt_prev
            
            # Wind speed evolution
            if alt < 100:  # Surface layer - strong shear
                # Power law with terrain roughness
                alpha = 0.25  # rough terrain
                speed_factor = (alt / max(alt_prev, 10.0)) ** alpha
                self.layer_speeds[i] = self.layer_speeds[i-1] * speed_factor
                # Add small random variation
                self.layer_speeds[i] *= (1.0 + self.rng.normal(0, 0.1))
                
            elif alt < 1000:  # Boundary layer - moderate increase
                # Logarithmic profile transitioning to free atmosphere
                speed_increase = self.rng.normal(0.5, 0.3) * (delta_alt / 100.0)
                self.layer_speeds[i] = self.layer_speeds[i-1] + speed_increase
                
            elif alt < 3000:  # Lower free atmosphere
                # Can have moderate wind speeds, some variability
                speed_change = self.rng.normal(0.2, 0.5) * (delta_alt / 500.0)
                self.layer_speeds[i] = self.layer_speeds[i-1] + speed_change
                
            else:  # Upper levels - potential jet stream effects
                # Higher variability, can increase significantly
                if self.rng.random() < 0.3:  # 30% chance of jet stream influence
                    speed_change = self.rng.normal(2.0, 1.5) * (delta_alt / 1000.0)
                else:
                    speed_change = self.rng.normal(0, 1.0) * (delta_alt / 1000.0)
                self.layer_speeds[i] = self.layer_speeds[i-1] + speed_change
            
            # Clamp speeds to reasonable values
            self.layer_speeds[i] = np.clip(self.layer_speeds[i], 0.5, 50.0)
            
            # Wind direction evolution (backing/veering with altitude)
            if alt < 100:  # Surface layer - friction effects
                # Direction can change significantly near surface
                dir_change = self.rng.normal(0, np.deg2rad(20)) * (delta_alt / 50.0)
                
            elif alt < 1000:  # Boundary layer - thermal wind effects
                # In Northern Hemisphere, typically veers (clockwise) with height
                # Add randomness to simulate different atmospheric conditions
                veer_rate = self.rng.normal(np.deg2rad(15), np.deg2rad(10))  # deg per 1000m
                dir_change = veer_rate * (delta_alt / 1000.0)
                
            elif alt < 3000:  # Lower free atmosphere
                # More variable, can back or veer
                dir_change = self.rng.normal(0, np.deg2rad(25)) * (delta_alt / 1000.0)
                
            else:  # Upper levels
                # Large-scale flow patterns, can shift significantly
                dir_change = self.rng.normal(0, np.deg2rad(30)) * (delta_alt / 1000.0)
            
            self.layer_directions[i] = self.layer_directions[i-1] + dir_change
            
            # Keep direction in [0, 2π]
            self.layer_directions[i] = self.layer_directions[i] % (2 * np.pi)

    def _get_turbulence_intensities(self, altitude: float) -> tuple[float, float, float]:
        """
        Get turbulence intensities (sigma_u, sigma_v, sigma_w) based on altitude
        and specified intensity level according to MIL-F-8785C.
        
        Returns: (sigma_u, sigma_v, sigma_w) in m/s
        """
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
            sigma_w = 0.1 * W20
            sigma_u = sigma_w / (0.177 + 0.000823 * h_ft) ** 0.4
            sigma_v = sigma_u
        else:  # Medium/high altitude
            sigma_w = 0.1 * W20
            sigma_u = sigma_w
            sigma_v = sigma_w
            
            # Decrease intensity with altitude
            if altitude > 1000:
                scale = max(0.1, 1.0 - (altitude - 1000) / 10000)
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
        
        if altitude < 304.8:  # Low altitude (< 1000 ft)
            L_w = altitude  # scale with altitude
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
        Get mean wind components at given altitude by interpolating through
        the pre-generated wind layers.
        """
        if altitude < 0:
            altitude = 0
        
        # Clamp to valid range
        alt = np.clip(altitude, self.layer_altitudes[0], self.layer_altitudes[-1])
        
        # Interpolate wind speed and direction
        wind_speed = np.interp(alt, self.layer_altitudes, self.layer_speeds)
        wind_direction = np.interp(alt, self.layer_altitudes, self.layer_directions)
        
        # Convert to components
        # Wind direction is "from" direction (meteorological convention)
        # North is 0°, East is 90°
        # Wind FROM north means blowing TO south (negative Y in NED)
        wind_x = -wind_speed * np.sin(wind_direction)  # East component
        wind_y = -wind_speed * np.cos(wind_direction)  # North component
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

    def update(self, t: float, altitude: float, airspeed: float) -> tuple[float, float, float, float]:
        """
        Update the Dryden turbulence model with layered winds.
        
        Args:
            t: Current time (s)
            altitude: Geometric altitude above ground (m), positive upwards
            airspeed: True airspeed magnitude (m/s)
        
        Returns:
            (density, wind_x, wind_y, wind_z)
        
        After calling, use self.DEN, self.VXWIND, self.VYWIND, self.VZWIND
        """
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
        
        # Get mean wind from layered profile
        mean_x, mean_y, mean_z = self._get_mean_wind(alt)
        
        # Total wind = mean + turbulence
        self.VXWIND = float(mean_x + u_gust)
        self.VYWIND = float(mean_y + v_gust)
        self.VZWIND = float(mean_z + w_gust)
        self.DEN = float(self._density_from_alt(alt))
        
        return self.DEN, self.VXWIND, self.VYWIND, self.VZWIND
    
    def get_wind_profile(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the wind profile for visualization/debugging.
        
        Returns:
            (altitudes, speeds, directions_deg)
        """
        return (self.layer_altitudes.copy(), 
                self.layer_speeds.copy(), 
                np.rad2deg(self.layer_directions))