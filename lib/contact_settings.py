import os
import numpy as np
from pathlib import Path
from typing import Union

from scipy.optimize import root
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

# --------------------------------------------------------- #
# Kalker's coefficients                                     #
# --------------------------------------------------------- #
c11tab = np.array(
    [
        3.31,
        3.37,
        3.44,
        3.53,
        3.62,
        3.72,
        3.81,
        3.91,
        4.01,
        4.12,
        4.22,
        4.36,
        4.54,
        4.78,
        5.10,
        5.57,
        6.34,
        7.78,
        11.7,
    ]
)
c22tab = np.array(
    [
        2.52,
        2.63,
        2.75,
        2.88,
        3.01,
        3.14,
        3.28,
        3.41,
        3.54,
        3.67,
        3.81,
        3.99,
        4.21,
        4.50,
        4.90,
        5.48,
        6.40,
        8.14,
        12.8,
    ]
)
c23tab = np.array(
    [
        0.47,
        0.60,
        0.72,
        0.82,
        0.93,
        1.03,
        1.14,
        1.25,
        1.36,
        1.47,
        1.59,
        1.75,
        1.95,
        2.23,
        2.62,
        3.24,
        4.32,
        6.63,
        14.6,
    ]
)
abratio = np.array(
    [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.11,
        1.25,
        1.43,
        1.67,
        2,
        2.5,
        3.33,
        5,
        10,
    ]
)