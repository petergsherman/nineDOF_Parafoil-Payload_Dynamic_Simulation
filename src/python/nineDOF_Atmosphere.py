from dataclasses import dataclass
from enum import Enum
import numpy as np


class TurbulenceMode(Enum):
    """Turbulence model selection"""
    NONE = "none"
    SIMPLE = "simple"
    DRYDEN = "dryden"


@dataclass
class staticAtmosphere:
    DEN: float = 1.22566     # Air density (kg/m^3)
    VXWIND: float = 0.0      # Wind velocity X (m/s)
    VYWIND: float = 0.0      # Wind velocity Y (m/s)
    VZWIND: float = 0.0      # Wind velocity Z (m/s)


class dynamicAtmosphere:
    """
    Simplified dynamic atmosphere with layered winds and selectable turbulence models.
    
    This model generates:
    - Layered mean wind profile with varying speed and direction by altitude
    - Selectable turbulence models:
      * NONE: No turbulence, just mean wind
      * SIMPLE: Smooth sinusoidal gusts with random direction and magnitude
      * DRYDEN: MIL-F-8785C Dryden turbulence model (continuous stochastic)
    
    Usage:
        atm = dynamicAtmosphere(turbulence_mode=TurbulenceMode.DRYDEN)
        ...
        altitude = -state[2]   # if z is down
        atm.update(t, altitude)
        # then use atm.DEN, atm.VXWIND, atm.VYWIND, atm.VZWIND
    """

    def __init__(
        self,
        turbulence_mode: TurbulenceMode = TurbulenceMode.SIMPLE,
        turbulence_intensity: str = "moderate",  # "light", "moderate", "severe"
        altitude_max: float = 10000.0,  # Maximum altitude for wind profile (m)
        n_layers: int = 20,  # Number of distinct wind layers
        seed: int | None = None,
    ) -> None:
        """
        Initialize simplified atmosphere model with layered winds and turbulence.
        
        Args:
            turbulence_mode: Type of turbulence (NONE, SIMPLE, DRYDEN)
            turbulence_intensity: Turbulence intensity for Dryden model
            altitude_max: Maximum altitude for wind layer generation (m)
            n_layers: Number of wind layers to generate
            seed: Random seed for reproducibility
        """
        self.turbulence_mode = turbulence_mode
        self.turbulence_intensity = turbulence_intensity
        self.altitude_max = altitude_max
        self.n_layers = n_layers
        
        # Random generator
        self.rng = np.random.default_rng(seed)
        
        # Generate random surface wind conditions
        self.surface_wind_speed = self.rng.uniform(0.0, 4.0)
        self.surface_wind_direction = self.rng.uniform(0.0, 360.0)
        
        # Public atmosphere state
        self.DEN: float = 1.22566
        self.VXWIND: float = 0.0
        self.VYWIND: float = 0.0
        self.VZWIND: float = 0.0
        
        # Generate layered wind profile
        self._generate_wind_layers()
        
        # Initialize turbulence model
        if self.turbulence_mode == TurbulenceMode.SIMPLE:
            self._init_simple_turbulence()
        elif self.turbulence_mode == TurbulenceMode.DRYDEN:
            self._init_dryden_turbulence()
        
        # Time tracking
        self._last_t: float | None = None

        # History storage for visualization
        self.hist_t = []
        self.hist_alt = []
        self.hist_gust_speed = []
        self.hist_gust_dir_deg = []


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

    # ==================== SIMPLE TURBULENCE MODEL ====================
    
    def _init_simple_turbulence(self) -> None:
        """Initialize simple sinusoidal gust model"""
        self._gust_active: bool = False
        self._gust_start_time: float = 0.0
        self._gust_duration: float = 0.0
        
        self._gust_peak_x: float = 0.0
        self._gust_peak_y: float = 0.0
        self._gust_peak_z: float = 0.0
        
        # Next gust timing - start after 10 second settling period
        self._next_gust_time: float = 10.0 + self.rng.uniform(10.0, 45.0)

    def _generate_new_gust(self) -> None:
        """Generate a new random gust with random direction and magnitude"""
        # Random gust magnitude between 1-15 m/s
        gust_magnitude = self.rng.uniform(1.0, 10.0)
        
        # Random direction (0-360 degrees)
        gust_direction = self.rng.uniform(0.0, 2 * np.pi)
        
        # Convert to components (horizontal plane)
        self._gust_peak_x = gust_magnitude * np.cos(gust_direction)
        self._gust_peak_y = gust_magnitude * np.sin(gust_direction)
        
        # Small vertical component
        self._gust_peak_z = self.rng.uniform(-2.0, 2.0)
        
        # Random gust duration between 3-8 seconds
        self._gust_duration = self.rng.uniform(3.0, 8.0)

    def _calculate_gust_component(self, t: float) -> tuple[float, float, float]:
        """Calculate current gust components using smooth sinusoidal profile"""
        if not self._gust_active:
            return 0.0, 0.0, 0.0
        
        elapsed = t - self._gust_start_time
        
        if elapsed >= self._gust_duration:
            self._gust_active = False
            return 0.0, 0.0, 0.0
        
        # Sinusoidal profile (half sine wave)
        phase = (elapsed / self._gust_duration) * np.pi
        amplitude = np.sin(phase)
        
        gust_x = self._gust_peak_x * amplitude
        gust_y = self._gust_peak_y * amplitude
        gust_z = self._gust_peak_z * amplitude
        
        return gust_x, gust_y, gust_z

    def _update_simple_turbulence(self, t: float) -> tuple[float, float, float]:
        """Update simple gust state and return current gust components"""
        # Check if it's time to start a new gust
        if not self._gust_active and t >= self._next_gust_time:
            self._generate_new_gust()
            self._gust_start_time = t
            self._gust_active = True
            
            # Schedule next gust
            self._next_gust_time = t + self._gust_duration + self.rng.uniform(5.0, 15.0)
        
        return self._calculate_gust_component(t)

    # ==================== DRYDEN TURBULENCE MODEL ====================
    
    def _init_dryden_turbulence(self) -> None:
        """
        Initialize Dryden turbulence model (MIL-F-8785C)
        
        Implements continuous stochastic turbulence using filtered white noise
        to match prescribed power spectral density functions.
        """
        # Turbulence intensity parameters (σ values in m/s)
        if self.turbulence_intensity == "light":
            self._turb_sigma_base = 1
        elif self.turbulence_intensity == "moderate":
            self._turb_sigma_base = 3
        else:  # severe
            self._turb_sigma_base = 5
        
        # Dryden filter states (for forming colored noise from white noise)
        self._dryden_state_u = 0.0  # Longitudinal (x) turbulence state
        self._dryden_state_v = 0.0  # Lateral (y) turbulence state
        self._dryden_state_w1 = 0.0  # Vertical (z) turbulence state 1
        self._dryden_state_w2 = 0.0  # Vertical (z) turbulence state 2
        
        # Current turbulence velocities
        self._turb_u = 0.0
        self._turb_v = 0.0
        self._turb_w = 0.0
    
    def _get_dryden_parameters(self, altitude: float, wind_speed_20ft: float) -> dict:
        """
        Calculate Dryden turbulence parameters based on altitude and wind.
        
        Based on MIL-F-8785C specifications for atmospheric turbulence.
        
        Args:
            altitude: Altitude above ground (m)
            wind_speed_20ft: Wind speed at 20 ft reference height (m/s)
        
        Returns:
            Dictionary with turbulence length scales and intensities
        """
        h = max(altitude, 0.3048)  # altitude in meters, min 1 ft
        
        # Convert to feet for MIL-SPEC calculations
        h_ft = h / 0.3048
        
        # Turbulence intensities (based on wind speed at 20 ft)
        W20 = max(wind_speed_20ft, 1.0)  # Wind speed at 20 ft (knots conversion)
        
        # Low altitude model (below 1000 ft)
        if h_ft < 1000:
            # Length scales (feet, then convert to meters)
            Lu_ft = h_ft / (0.177 + 0.000823 * h_ft) ** 1.2
            Lv_ft = Lu_ft / 2.0
            Lw_ft = h_ft
            
            # Turbulence intensities (ft/s)
            sigma_w_fps = 0.1 * W20
            sigma_u_fps = sigma_w_fps / (0.177 + 0.000823 * h_ft) ** 0.4
            sigma_v_fps = sigma_u_fps
            
        # High altitude model (1000 ft and above)
        else:
            # Length scales (feet)
            Lu_ft = 1750.0
            Lv_ft = 1750.0
            Lw_ft = 1750.0
            
            # Turbulence intensities (ft/s) - decrease with altitude
            sigma_w_fps = 0.1 * W20
            sigma_u_fps = sigma_w_fps
            sigma_v_fps = sigma_w_fps
        
        # Convert to SI units (meters, m/s)
        Lu = Lu_ft * 0.3048
        Lv = Lv_ft * 0.3048
        Lw = Lw_ft * 0.3048
        
        # Apply intensity scaling
        sigma_u = sigma_u_fps * 0.3048 * self._turb_sigma_base
        sigma_v = sigma_v_fps * 0.3048 * self._turb_sigma_base
        sigma_w = sigma_w_fps * 0.3048 * self._turb_sigma_base
        
        return {
            'Lu': Lu, 'Lv': Lv, 'Lw': Lw,
            'sigma_u': sigma_u, 'sigma_v': sigma_v, 'sigma_w': sigma_w
        }
    
    def _update_dryden_turbulence(self, dt: float, altitude: float, airspeed: float) -> tuple[float, float, float]:
        """
        Update Dryden turbulence model using first-order filter approximation.
        
        Args:
            dt: Time step (s)
            altitude: Altitude above ground (m)
            airspeed: True airspeed (m/s)
        
        Returns:
            (turb_u, turb_v, turb_w) turbulence velocity components
        """
        if dt <= 0 or dt > 1.0:  # Safety check
            dt = 0.01
        
        # Get current wind speed for turbulence scaling
        mean_wind_speed = np.sqrt(self.VXWIND**2 + self.VYWIND**2)
        wind_speed_20ft = max(mean_wind_speed, 1.0)
        
        # Get Dryden parameters
        params = self._get_dryden_parameters(altitude, wind_speed_20ft)
        
        # Use airspeed for temporal scaling (frozen turbulence hypothesis)
        V = max(airspeed, 5.0)  # Minimum 5 m/s to avoid division issues
        
        # Time constants for each axis (tau = L/V)
        tau_u = params['Lu'] / V
        tau_v = params['Lv'] / V
        tau_w = params['Lw'] / V
        
        # Filter coefficients (first-order approximation)
        # dx/dt = -x/tau + sigma*sqrt(2/tau)*w(t)
        # Discrete: x[k+1] = a*x[k] + b*w[k]
        a_u = np.exp(-dt / tau_u)
        a_v = np.exp(-dt / tau_v)
        a_w = np.exp(-dt / tau_w)
        
        b_u = params['sigma_u'] * np.sqrt(1 - a_u**2)
        b_v = params['sigma_v'] * np.sqrt(1 - a_v**2)
        b_w = params['sigma_w'] * np.sqrt(1 - a_w**2)
        
        # White noise inputs
        noise_u = self.rng.standard_normal()
        noise_v = self.rng.standard_normal()
        noise_w = self.rng.standard_normal()
        
        # Update filter states (first-order)
        self._dryden_state_u = a_u * self._dryden_state_u + b_u * noise_u
        self._dryden_state_v = a_v * self._dryden_state_v + b_v * noise_v
        self._dryden_state_w1 = a_w * self._dryden_state_w1 + b_w * noise_w
        
        # Output turbulence velocities
        self._turb_u = self._dryden_state_u
        self._turb_v = self._dryden_state_v
        self._turb_w = self._dryden_state_w1
        
        return self._turb_u, self._turb_v, self._turb_w

    # ==================== COMMON METHODS ====================

    def _get_mean_wind(self, altitude: float) -> tuple[float, float, float]:
        """Get mean wind components at given altitude by interpolating layers"""
        if altitude < 0:
            altitude = 0
        
        alt = np.clip(altitude, self.layer_altitudes[0], self.layer_altitudes[-1])
        
        # Interpolate wind speed and direction
        wind_speed = np.interp(alt, self.layer_altitudes, self.layer_speeds)
        wind_direction = np.interp(alt, self.layer_altitudes, self.layer_directions)
        
        # Convert to components
        wind_x = -wind_speed * np.sin(wind_direction)  # East component
        wind_y = -wind_speed * np.cos(wind_direction)  # North component
        wind_z = 0.0
        
        return wind_x, wind_y, wind_z

    def _density_from_alt(self, altitude: float) -> float:
        """Standard atmosphere density model"""
        rho0 = 1.22566  # kg/m^3 at sea level
        H = 8500.0      # scale height [m]
        alt_clamped = max(0.0, altitude)
        return rho0 * np.exp(-alt_clamped / H)

    def update(self, t: float, altitude: float, airspeed: float = 20.0) -> tuple[float, float, float, float]:
        """
        Update the atmosphere model with layered winds and turbulence.
        
        Args:
            t: Current time (s)
            altitude: Geometric altitude above ground (m), positive upwards
            airspeed: True airspeed for Dryden model (m/s), optional
        
        Returns:
            (density, wind_x, wind_y, wind_z)
        
        After calling, use self.DEN, self.VXWIND, self.VYWIND, self.VZWIND
        """
        alt = float(max(0.0, altitude))
        
        # Calculate time step
        if self._last_t is None:
            dt = 0.01
        else:
            dt = t - self._last_t
        self._last_t = t
        
        # Get mean wind from layered profile
        mean_x, mean_y, mean_z = self._get_mean_wind(alt)
        
        # Get turbulence based on selected mode
        turb_x, turb_y, turb_z = 0.0, 0.0, 0.0
        
        if self.turbulence_mode == TurbulenceMode.SIMPLE:
            turb_x, turb_y, turb_z = self._update_simple_turbulence(t)
        elif self.turbulence_mode == TurbulenceMode.DRYDEN:
            turb_x, turb_y, turb_z = self._update_dryden_turbulence(dt, alt, airspeed)
        # NONE mode: turbulence stays at 0.0
        
        # Total wind = mean + turbulence
        self.VXWIND = float(mean_x + turb_x)
        self.VYWIND = float(mean_y + turb_y)
        self.VZWIND = float(mean_z + turb_z)
        self.DEN = float(self._density_from_alt(alt))

        # -----------------------------
        # Gust history logging
        # -----------------------------
        gust_speed = np.sqrt(turb_x**2 + turb_y**2)
        gust_dir_deg = np.degrees(np.arctan2(turb_y, turb_x))  # [-180, 180]

        self.hist_t.append(t)
        self.hist_alt.append(altitude)
        self.hist_gust_speed.append(gust_speed)
        self.hist_gust_dir_deg.append(gust_dir_deg)

        
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
    
    def get_turbulence_info(self) -> dict:
        """
        Get current turbulence information for debugging.
        
        Returns:
            Dictionary with turbulence state information
        """
        info = {'mode': self.turbulence_mode.value}
        
        if self.turbulence_mode == TurbulenceMode.SIMPLE:
            if self._gust_active:
                elapsed = self._last_t - self._gust_start_time if self._last_t else 0.0
                phase = (elapsed / self._gust_duration) * np.pi if self._gust_duration > 0 else 0.0
                amplitude = np.sin(phase)
            else:
                elapsed = 0.0
                amplitude = 0.0
            
            info.update({
                'active': self._gust_active,
                'elapsed': elapsed,
                'duration': self._gust_duration,
                'amplitude': amplitude,
                'peak_magnitude': np.sqrt(self._gust_peak_x**2 + self._gust_peak_y**2 + self._gust_peak_z**2),
                'next_gust_in': max(0.0, self._next_gust_time - (self._last_t or 0.0))
            })
        
        elif self.turbulence_mode == TurbulenceMode.DRYDEN:
            info.update({
                'intensity': self.turbulence_intensity,
                'turb_u': self._turb_u,
                'turb_v': self._turb_v,
                'turb_w': self._turb_w,
                'turb_magnitude': np.sqrt(self._turb_u**2 + self._turb_v**2 + self._turb_w**2)
            })
        
        return info