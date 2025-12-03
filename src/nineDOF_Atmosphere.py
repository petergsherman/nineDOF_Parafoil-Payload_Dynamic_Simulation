#nineDOF_Atmosphere.py

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
    def __init__(self) -> None:
        pass