from dataclasses import dataclass
import numpy as np


@dataclass
class staticAtmosphere:
    DEN: float = 1.22566     # Air density (kg/m^3)
    VXWIND: float = 0.0      # Wind velocity X (m/s)
    VYWIND: float = 0.0      # Wind velocity Y (m/s)
    VZWIND: float = 0.0      # Wind velocity Z (m/s)


class dynamicAtmosphere:
    """
    Simplified dynamic atmosphere with layered winds and smooth sinusoidal gusts.
    
    This model generates:
    - Layered mean wind profile with varying speed and direction by altitude
    - Smooth sinusoidal wind gusts that build up and decay gradually
    - Gusts have random direction (0-360°) and peak magnitude (1-15 m/s)
    - Natural acceleration profiles that avoid numerical instability
    
    Usage:
        atm = dynamicAtmosphere(gust_enabled=True)
        ...
        altitude = -state[2]   # if z is down
        atm.update(t, altitude)
        # then use atm.DEN, atm.VXWIND, atm.VYWIND, atm.VZWIND
    """

    def __init__(
        self,
        gust_enabled: bool = True,
        altitude_max: float = 10000.0,  # Maximum altitude for wind profile (m)
        n_layers: int = 20,  # Number of distinct wind layers
        seed: int | None = None,
    ) -> None:
        """
        Initialize simplified atmosphere model with layered winds and sinusoidal gusts.
        
        Args:
            gust_enabled: Enable random wind gusts
            altitude_max: Maximum altitude for wind layer generation (m)
            n_layers: Number of wind layers to generate
            seed: Random seed for reproducibility
        """
        self.gust_enabled = gust_enabled
        self.altitude_max = altitude_max
        self.n_layers = n_layers
        
        # Random generator
        self.rng = np.random.default_rng(seed)
        
        # Generate random surface wind conditions
        self.surface_wind_speed = self.rng.uniform(2.0, 12.0)
        self.surface_wind_direction = self.rng.uniform(0.0, 360.0)
        
        # Public atmosphere state
        self.DEN: float = 1.22566
        self.VXWIND: float = 0.0
        self.VYWIND: float = 0.0
        self.VZWIND: float = 0.0
        
        # Generate layered wind profile
        self._generate_wind_layers()
        
        # Gust state variables - sinusoidal approach
        self._gust_active: bool = False
        self._gust_start_time: float = 0.0
        self._gust_duration: float = 0.0  # Total duration of gust (ramp up + hold + ramp down)
        
        self._gust_peak_x: float = 0.0
        self._gust_peak_y: float = 0.0
        self._gust_peak_z: float = 0.0
        
        # Next gust timing - start after 10 second settling period
        self._next_gust_time: float = 10.0 + self.rng.uniform(10.0, 45.0) if gust_enabled else float('inf')
        
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

    def _generate_new_gust(self) -> None:
        """
        Generate a new random gust with random direction and magnitude.
        Sets the peak values that will be reached via sinusoidal buildup.
        """
        # Random gust magnitude between 1-15 m/s
        gust_magnitude = self.rng.uniform(1.0, 10.0)
        
        # Random direction (0-360 degrees)
        gust_direction = self.rng.uniform(0.0, 2 * np.pi)
        
        # Convert to components (horizontal plane)
        self._gust_peak_x = gust_magnitude * np.cos(gust_direction)
        self._gust_peak_y = gust_magnitude * np.sin(gust_direction)
        
        # Small vertical component (typically much smaller than horizontal)
        self._gust_peak_z = self.rng.uniform(-2.0, 2.0)
        
        # Random gust duration between 3-8 seconds (time from start to finish)
        self._gust_duration = self.rng.uniform(3.0, 8.0)

    def _calculate_gust_component(self, t: float) -> tuple[float, float, float]:
        """
        Calculate current gust components using smooth sinusoidal profile.
        
        The gust follows a sine wave pattern:
        - Starts at 0
        - Smoothly ramps up to peak
        - Smoothly ramps back down to 0
        
        This creates smooth accelerations that match real atmospheric behavior.
        """
        if not self._gust_active:
            return 0.0, 0.0, 0.0
        
        # Time since gust started
        elapsed = t - self._gust_start_time
        
        # Check if gust is complete
        if elapsed >= self._gust_duration:
            self._gust_active = False
            return 0.0, 0.0, 0.0
        
        # Calculate sinusoidal profile (half sine wave from 0 to π)
        # This gives smooth acceleration at start and deceleration at end
        phase = (elapsed / self._gust_duration) * np.pi
        amplitude = np.sin(phase)  # Ranges from 0 to 1 and back to 0
        
        # Apply amplitude to peak gust components
        gust_x = self._gust_peak_x * amplitude
        gust_y = self._gust_peak_y * amplitude
        gust_z = self._gust_peak_z * amplitude
        
        return gust_x, gust_y, gust_z

    def _update_gust(self, t: float) -> tuple[float, float, float]:
        """
        Update gust state and return current gust components.
        """
        if not self.gust_enabled:
            return 0.0, 0.0, 0.0
        
        # Check if it's time to start a new gust
        if not self._gust_active and t >= self._next_gust_time:
            # Start new gust
            self._generate_new_gust()
            self._gust_start_time = t
            self._gust_active = True
            
            # Schedule next gust (5-15 seconds after current gust completes)
            self._next_gust_time = t + self._gust_duration + self.rng.uniform(5.0, 15.0)
        
        # Calculate and return current gust components
        return self._calculate_gust_component(t)

    def update(self, t: float, altitude: float) -> tuple[float, float, float, float]:
        """
        Update the atmosphere model with layered winds and smooth sinusoidal gusts.
        
        Args:
            t: Current time (s)
            altitude: Geometric altitude above ground (m), positive upwards
        
        Returns:
            (density, wind_x, wind_y, wind_z)
        
        After calling, use self.DEN, self.VXWIND, self.VYWIND, self.VZWIND
        """
        # Clamp altitude
        alt = float(max(0.0, altitude))
        
        # Time step (for reference, not used in sinusoidal approach)
        if self._last_t is None:
            dt = 0.01
        else:
            dt = t - self._last_t
        self._last_t = t
        
        # Get mean wind from layered profile
        mean_x, mean_y, mean_z = self._get_mean_wind(alt)
        
        # Get current gust components (smooth sinusoidal)
        gust_x, gust_y, gust_z = self._update_gust(t)
        
        # Total wind = mean + gust
        self.VXWIND = float(mean_x + gust_x)
        self.VYWIND = float(mean_y + gust_y)
        self.VZWIND = float(mean_z + gust_z)
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
    
    def get_current_gust_info(self) -> dict:
        """
        Get current gust information for debugging.
        
        Returns:
            Dictionary with gust state information
        """
        if self._gust_active:
            elapsed = self._last_t - self._gust_start_time if self._last_t else 0.0
            phase = (elapsed / self._gust_duration) * np.pi if self._gust_duration > 0 else 0.0
            amplitude = np.sin(phase)
        else:
            elapsed = 0.0
            amplitude = 0.0
        
        return {
            'active': self._gust_active,
            'elapsed': elapsed,
            'duration': self._gust_duration,
            'amplitude': amplitude,
            'peak_magnitude': np.sqrt(self._gust_peak_x**2 + self._gust_peak_y**2 + self._gust_peak_z**2),
            'next_gust_in': max(0.0, self._next_gust_time - (self._last_t or 0.0))
        }