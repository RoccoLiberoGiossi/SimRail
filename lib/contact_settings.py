import os
import numpy as np
from pathlib import Path
from typing import Union

from scipy.optimize import root, brentq
from scipy.integrate import simpson
from scipy import interpolate
from scipy.interpolate import PchipInterpolator

# --------------------------------------------------------- #
# Global parameters definition                              #
# --------------------------------------------------------- #
LEFT = 0
RIGHT = 1

# --------------------------------------------------------- #
# Correction coefficients for contact patch dimentions      #
# --------------------------------------------------------- #
EQU_EL_POWER_CORRECTION = 1.026600611713163413e00
EQU_EL_DIV_CORRECTION = 1.397839912270349316e00

# --------------------------------------------------------- #
# GLobal files definition                                   #
# --------------------------------------------------------- #
root_pwd = Path.cwd()
profiles_path = root_pwd.joinpath("Profiles")
wheel_path = profiles_path.joinpath("Wheel_Profiles" + os.sep + "S1002.txt")
rail_path = profiles_path.joinpath("Rail_Profiles" + os.sep + "UIC60.txt")
S1002_profile = np.loadtxt(wheel_path, delimiter=",")
UIC60_profile = np.loadtxt(rail_path, delimiter=",")