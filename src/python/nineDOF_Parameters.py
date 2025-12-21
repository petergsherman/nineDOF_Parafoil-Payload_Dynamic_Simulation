#9DOF_Parameters.py
import numpy as np 
from dataclasses import dataclass

@dataclass
class systemParameters:

    #Masses and Geometry
    m_parafoil:     float = 8.99 #Parafoil mass (kg)
    m_cradle:       float = 90.0 #Cradle mass (kg)
    A_parafoil:     float = 27.0 #Parafoil area (m^2)
    A_cradle:       float = 0.4337 #Cradle area (m^2)
    cbar_parafoil:  float = 3.87 #Parafoil mean chord (m)
    bbar_parafoil:  float = 6.99 #Parafoil span (m)
    dbar:           float = 1.0 #Brake deflection reference
    deadband:       float = 0.0 #Control deadband
    CD_cradle:      float = 1.0 #Cradle drag coefficient

    #Parafoil Inertia (kg-m^2)
    PIXX: float = 74.56
    PIXY: float = 0.0
    PIXZ: float = 0.0
    PIYY: float = 14.62
    PIYZ: float = 0.0
    PIZZ: float = 82.8
    
    #Cradle Inertia (kg-m^2)
    CIXX: float = 9.378
    CIXY: float = 0.0
    CIXZ: float = 0.0
    CIYY: float = 6.0518
    CIYZ: float = 0.0
    CIZZ: float = 6.2401

    #Geometry vectors (m) SL = Stationline, BL = Buttline, WL = Waterline
    SLCRADLE: float = 0.0    #Gimbal to cradle CG
    BLCRADLE: float = 0.0
    WLCRADLE: float = 0.47
    
    SLPARAFOIL: float = 0.0  #Gimbal to parafoil CG
    BLPARAFOIL: float = 0.0
    WLPARAFOIL: float = -7.622
    
    SLPR: float = 0.0        #Parafoil CG to rotation point
    BLPR: float = 0.0
    WLPR: float = 0.0
    
    SLRAP: float = 0.0       #Rotation point to aero center (in incidence frame)
    BLRAP: float = 0.0
    WLRAP: float = 0.0
    
    SLPMP: float = 0.0       #Parafoil CG to apparent mass center
    BLPMP: float = 0.0
    WLPMP: float = 7.622

    #Apparent Mass and Inertia Coefficients (kg, kg-m)
    PMASSA: float = 1.05
    PMASSB: float = 6.46
    PMASSÇ: float = 31.78
    PMASSH: float = 0.0
    PMASSP: float = 18.36
    PMASSQ: float = 26.5
    PMASSR: float = 7.104
    
    #Gimbal Properties
    KGIMBAL: float = 16.244  #Rotational Stiffness (N-m/rad)
    CGIMBAL: float = 1.3537  #Rotational Damping (N-m/(rad/s))
    
    #Control parameters
    NOM_INCIDENCE: float = -0.055  #Nominal incidence angle (rad)
    NOM_BRAKE: float = 0.5         #Nominal brake deflection (m)
    TAU: float = 0.6              #Brake time constant (s)
    
    #Aerodynamic coefficients (single-point model)
    CM0: float = 0.0
    CMQ: float = -1.0
    CMDS: float = 0.0
    CYBETA: float = -1.0
    CNBETA: float = 0.0
    CLBETA: float = 0.0
    CLP: float = 0.112
    CLR: float = 0.253
    CLDA: float = 0.0
    CNP: float = -0.050
    CNR: float = -0.212
    CND1: float = 0.0

    #Aerodynamic tables (sig1 interpolation)
    SIGTAB: np.ndarray = None
    CD0TAB: np.ndarray = None
    CDA2TAB: np.ndarray = None
    CL0TAB: np.ndarray = None
    CLATAB: np.ndarray = None
    
    #AOA table
    AOATAB: np.ndarray = None
    CND2TAB: np.ndarray = None
    
    #Physical constants
    GRAVITY: float = 9.81    # m/s^2

    def __post_init__(self):
            """Initialize aerodynamic tables from input file data"""
            if self.SIGTAB is None:
                # Default tables from input file
                self.SIGTAB = np.array([0.2500, 1.5000])
                self.CD0TAB = np.array([0.1373, 0.0351])
                self.CDA2TAB = np.array([0.0977, 3.9642])
                self.CL0TAB = np.array([-0.2636, 0.1313])
                self.CLATAB = np.array([2.8139, 1.9825])
                
            if self.AOATAB is None:
               self.AOATAB = np.array([0.1800, 0.2500, 0.3200])
               self.CND2TAB = np.array([0.0280, 0.0350, 0.0700])


@dataclass
class atmosphereParameters:
    """Atmosphere model parameters"""
    DEN: float = 1.22566     # Air density (kg/m^3)
    VXWIND: float = 0.0      # Wind velocity X (m/s)
    VYWIND: float = 0.0      # Wind velocity Y (m/s)
    VZWIND: float = 0.0      # Wind velocity Z (m/s)


# ============================================================================
# STATE VECTOR LAYOUT (18 states total)
# ============================================================================
# States 0-2:   xg, yg, zg          - Gimbal position in inertial frame (m)
# States 3-5:   phip, thetap, psip  - Parafoil Euler angles (rad)
# States 6-8:   phic, thetac, psic  - Cradle Euler angles (rad)
# States 9-11:  ug, vg, wg          - Gimbal velocity in parafoil frame (m/s)
# States 12-14: pp, qp, rp          - Parafoil angular rates (rad/s)
# States 15-17: pc, qc, rc          - Cradle angular rates (rad/s)
# ============================================================================
