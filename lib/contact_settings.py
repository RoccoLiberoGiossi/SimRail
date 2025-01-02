import os
import numpy as np
from pathlib import Path

from scipy.optimize import fsolve
from scipy import interpolate

# --------------------------------------------------------- #
# GLobal parameters definition                              #
# --------------------------------------------------------- #
LEFT = 0
RIGHT = 1

EQU_EL_POWER_CORRECTION = 1.026600611713163413e00
EQU_EL_DIV_CORRECTION = 1.397839912270349316e00

# --------------------------------------------------------- #
# GLobal files definition                                   #
# --------------------------------------------------------- #
root = Path.cwd()
profiles_path = root.joinpath("Profiles")
wheel_path = profiles_path.joinpath("Wheel_Profiles"+os.sep+"S1002.txt")
rail_path = profiles_path.joinpath("Rail_Profiles"+os.sep+"UIC60.txt")
S1002_profile = np.loadtxt(wheel_path, delimiter=",")
UIC60_profile = np.loadtxt(rail_path, delimiter=",")

# class datawheel:
#     def __init__(
#             self, Dy = 0, Appr = 0, NF = 0, WA = 0, QA = 0, DZ = 0, Cent = 0,DyWheel = 0, DzWheel = 0,
#             aH = 0, bH = 0, AHertz = 0, BHertz = 0, rWy = 0, rWx = 0, rRy = 0,
#             ratioHertz = 0, tetaHertz = 0, DyRail = 0, DzRail = 0, StartPos = 0, EndPos = 0, Radius = 0
#     ):
#         self.DyWheeslet = Dy
#         self.approach = Appr # Done
#         self.NormalForce = NF # Done
#         self.WheelAngle = WA
#         self.QForce = QA # Done
#         self.Deltaz = DZ
#         self.centroid = Cent # Done
#         self.DyWheel = DyWheel
#         self.DzWheel = DzWheel
#         self.a = aH # Done
#         self.b = bH # Done
#         self.AHertz = AHertz # Done
#         self.BHertz = BHertz # Done
#         self.rWy = rWy
#         self.rWx = rWx
#         self.rRy = rRy
#         self.tetaHertz = tetaHertz # Done
#         self.ratioHertz = ratioHertz # Done
#         self.DyRail = DyRail
#         self.DzRail = DzRail
#         self.StartPos = StartPos
#         self.EndPos = EndPos
#         self.Radius = Radius

#     def reset_to_zero(self):
#         for attribute in vars(self):
#             setattr(self, attribute, 0)
