from dataclasses import dataclass
import numpy as np
import random 

@dataclass
class staticAtmosphere:
    DEN: float = 1.22566     # Air density (kg/m^3)
    VXWIND: float = 0.0      # Wind velocity X (m/s)
    VYWIND: float = 0.0      # Wind velocity Y (m/s)
    VZWIND: float = 0.0      # Wind velocity Z (m/s)


class dynamicAtmosphere:
    """
    Dynamic atmosphere model with:
      - Smooth horizontal wind variation with altitude (0–100 km)
      - Time-varying small-scale turbulence
      - Intermittent gust events with realistic durations and magnitudes
      - Simple exponential density model

    Usage pattern in a sim loop:
        atm = dynamicAtmosphere()
        ...
        altitude = -state[2]   # if z is down
        atm.update(t, altitude)
        # then use atm.DEN, atm.VXWIND, atm.VYWIND, atm.VZWIND
    """

    def __init__(
        self,
        alt_min: float = 0.0,
        alt_max: float = 100_000.0,  # 100 km
        n_layers: int = 201,
        gust_rate: float = 1.0 / 30.0,  # mean ~1 gust every 30 s
        gust_min_duration: float = 3.0,
        gust_max_duration: float = 10.0,
        bg_tau: float = 5.0,     # correlation time for background turbulence [s]
        bg_sigma: float = 1.0,   # base std dev of background turbulence [m/s]
        seed: int | None = None,
    ) -> None:
        self.alt_min = alt_min
        self.alt_max = alt_max
        self.n_layers = n_layers

        self.gust_rate = gust_rate
        self.gust_min_duration = gust_min_duration
        self.gust_max_duration = gust_max_duration

        self.bg_tau = bg_tau
        self.bg_sigma = bg_sigma

        # Public "current" atmosphere state (like staticAtmosphere)
        self.DEN: float = 1.22566
        self.VXWIND: float = 0.0
        self.VYWIND: float = 0.0
        self.VZWIND: float = 0.0

        # Random generator
        self.rng = np.random.default_rng(seed)

        # Precomputed baseline profile vs altitude
        self.alt_grid = np.linspace(self.alt_min, self.alt_max, self.n_layers)
        self.u_profile, self.v_profile, self.w_profile = self._build_baseline_profile()

        # Background small-scale turbulence (Ornstein–Uhlenbeck-like)
        self._bg_wind = np.zeros(3)

        # Gust event state
        self._gust_active = False
        self._gust_start = 0.0
        self._gust_peak = 0.0
        self._gust_end = 0.0
        self._gust_vector = np.zeros(3)

        # Time tracking
        self._last_t: float | None = None

    # ------------------------------------------------------------------
    # Baseline profile construction
    # ------------------------------------------------------------------
    def _mean_wind_speed(self, alt: float) -> float:
        """
        Crude "realistic-ish" mean wind-speed profile vs altitude.
        alt in meters (0–100,000).
        """
        if alt < 1_000.0:
            # Near ground: 0–1 km
            return 5.0 + 3.0 * (alt / 1_000.0)
        elif alt < 5_000.0:
            # 1–5 km: stronger low-level winds
            return 8.0 + 4.0 * (alt - 1_000.0) / 4_000.0
        elif alt < 12_000.0:
            # 5–12 km: jet-stream region
            return 15.0 + 20.0 * (alt - 5_000.0) / 7_000.0
        elif alt < 20_000.0:
            # 12–20 km: taper off
            return 35.0 - 15.0 * (alt - 12_000.0) / 8_000.0
        elif alt < 50_000.0:
            # 20–50 km: weaker stratospheric winds
            return 10.0 - 5.0 * (alt - 20_000.0) / 30_000.0
        else:
            # 50–100 km: very low density, small effective winds
            return 5.0

    def _build_baseline_profile(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a smooth baseline wind profile (u,v,w) over altitude.
        Direction changes smoothly with height; magnitude follows
        _mean_wind_speed plus some randomness.
        """
        dz = (self.alt_max - self.alt_min) / max(self.n_layers - 1, 1)

        # Wind direction as a random walk in altitude (smooth turning)
        direction = np.zeros(self.n_layers)
        direction[0] = self.rng.uniform(0.0, 2.0 * np.pi)

        for i in range(1, self.n_layers):
            # Small change in direction per layer (std ~ 15 deg over ~1 km)
            std_rad = np.deg2rad(15.0) * np.sqrt(dz / 1_000.0)
            direction[i] = direction[i - 1] + self.rng.normal(0.0, std_rad)

        # Wind speed magnitude with randomness around the mean profile
        speed = np.zeros(self.n_layers)
        for i, alt in enumerate(self.alt_grid):
            mean = self._mean_wind_speed(alt)
            # Lognormal-ish variation, clamped
            noise = self.rng.normal(0.0, 0.3 * mean)
            speed[i] = max(0.0, mean + noise)

        # Convert to components; vertical baseline very small
        u = speed * np.cos(direction)
        v = speed * np.sin(direction)
        w = self.rng.normal(0.0, 0.2, size=self.n_layers)  # small vertical component

        return u, v, w

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _interp_baseline(self, alt: float) -> np.ndarray:
        """
        Interpolate the baseline wind at given altitude.
        Returns np.array([u, v, w]).
        """
        alt_clamped = np.clip(alt, self.alt_min, self.alt_max)
        u = np.interp(alt_clamped, self.alt_grid, self.u_profile)
        v = np.interp(alt_clamped, self.alt_grid, self.v_profile)
        w = np.interp(alt_clamped, self.alt_grid, self.w_profile)
        return np.array([u, v, w])

    def _gust_scale(self, alt: float) -> float:
        """
        Scale factor for gust/background intensity vs altitude.
        Strongest in the lower ~10 km.
        """
        if alt <= 10_000.0:
            return 1.0 + 0.5 * (1.0 - alt / 10_000.0)  # up to 1.5 near ground
        elif alt <= 30_000.0:
            return 0.8
        else:
            return 0.5

    def _density_from_alt(self, alt: float) -> float:
        """
        Simple exponential standard atmosphere approximation.
        rho = rho0 * exp(-alt / H)
        """
        rho0 = 1.22566  # kg/m^3 at sea level
        H = 8_500.0     # scale height [m]
        alt_clamped = max(0.0, alt)
        return rho0 * np.exp(-alt_clamped / H)

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------
    def update(self, t: float, altitude: float) -> tuple[float, float, float, float]:
        """
        Update the dynamic atmosphere state for time t [s] and altitude [m].

        altitude: geometric altitude above "ground" (0 m), positive upwards.
                  For your sim where z is downwards, you probably want:
                      altitude = -state[2]

        After calling this, use:
            self.DEN, self.VXWIND, self.VYWIND, self.VZWIND
        """
        # Clamp altitude to modeled range
        alt = float(np.clip(altitude, self.alt_min, self.alt_max))

        # Time step
        if self._last_t is None:
            dt = 0.0
        else:
            dt = max(1e-3, t - self._last_t)
        self._last_t = t

        # --- Baseline wind vs altitude ---
        base_wind = self._interp_baseline(alt)

        # --- Background small-scale turbulence (OU-like) ---
        if dt > 0.0:
            tau = self.bg_tau
            decay = np.exp(-dt / tau)
            # Effective sigma scaled by altitude
            sigma = self.bg_sigma * self._gust_scale(alt)
            # This approximates a stationary OU process
            self._bg_wind = (decay * self._bg_wind +
                             np.sqrt(1.0 - decay ** 2) * sigma * self.rng.normal(size=3))

        # --- Gust event logic ---
        gust_vec = np.zeros(3)

        # End existing gust if needed
        if self._gust_active and t >= self._gust_end:
            self._gust_active = False

        # Possibly start a new gust event
        if dt > 0.0 and not self._gust_active:
            # Poisson process for gust start
            p_gust = 1.0 - np.exp(-self.gust_rate * dt)
            if self.rng.random() < p_gust:
                self._gust_active = True
                duration = self.rng.uniform(self.gust_min_duration, self.gust_max_duration)
                self._gust_start = t
                self._gust_end = t + duration
                self._gust_peak = t + 0.3 * duration  # peak early in the gust

                # Horizontal gust direction and magnitude
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                # Gust magnitude scaled with altitude & randomness
                base_mag = self._gust_scale(alt) * self.rng.uniform(5.0, 15.0)
                gust_u = base_mag * np.cos(angle)
                gust_v = base_mag * np.sin(angle)
                gust_w = self.rng.normal(0.0, 1.0)  # small vertical component
                self._gust_vector = np.array([gust_u, gust_v, gust_w])

        # If a gust is active, compute its envelope
        if self._gust_active:
            if t <= self._gust_peak:
                # Ramp up
                denom = max(self._gust_peak - self._gust_start, 1e-6)
                factor = (t - self._gust_start) / denom
            else:
                # Decay
                denom = max(self._gust_end - self._gust_peak, 1e-6)
                factor = max(0.0, (self._gust_end - t) / denom)

            gust_vec = factor * self._gust_vector

        # --- Total wind vector ---
        total_wind = base_wind #+ self._bg_wind + gust_vec

        # Update public attributes
        self.VXWIND = float(total_wind[0])
        self.VYWIND = float(total_wind[1])
        self.VZWIND = float(total_wind[2])
        self.DEN = float(self._density_from_alt(alt))

        return self.DEN, self.VXWIND, self.VYWIND, self.VZWIND
