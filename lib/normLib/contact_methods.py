import numpy as np
import matplotlib.pyplot as plt

from lib.contact_settings import *
from lib.contact_material import material
from lib.geoLib.contact_wheel_rail import wheel, rail
from lib.jittable_functions import interpolator

class normal_contact:
    def __init__(
        self,
        approach=0,
        normal_force=0,
        centroid_wheel=0,
        centroid_rail=0,
        Q_force=0,
        Y_force=0,
        contact_angle=0,
    ):
        self.approach = approach
        self.normal_force = normal_force
        self.Q_force = Q_force
        self.Y_force = Y_force
        self.centroid_wheel = centroid_wheel
        self.centroid_rail = centroid_rail
        self.contact_angle = contact_angle

    def reset_to_zero(self):
        for attribute in vars(self):
            setattr(self, attribute, 0)

# ----------------------------------------------------------------- #
# Equivalent Contact Method                                         #
# ----------------------------------------------------------------- #
class eqv_el_normal_contact(normal_contact):
    def __init__(
        self,
        semi_axis_a=0,
        semi_axis_b=0,
        A_Hertz=0,
        B_Hertz=0,
        teta_Hertz=0,
        ratio_Hertz=0,
        Rlocal=0,
    ):
        super().__init__()
        self.semi_axis_a = semi_axis_a
        self.semi_axis_b = semi_axis_b
        self.A_Hertz = A_Hertz
        self.B_Hertz = B_Hertz
        self.teta_Hertz = teta_Hertz
        self.ratio_Hertz = ratio_Hertz
        self.Rlocal = Rlocal

    def reset_to_zero(self):
        for attribute in vars(self):
            setattr(self, attribute, 0)

    @staticmethod
    def findAlpha(alpha, Area, Width):
        Res = Area - 0.5 * Width**2 / 4 * (alpha - np.sin(alpha)) / (np.sin(alpha / 2) ** 2)
        return Res

    def calculate_contact(
        self,
        LocalX: np.ndarray[float],
        Shape: np.ndarray[float],
        Rlocal: np.ndarray[float],
        material_model: material,
    ):
        # discretization is used in the kik_pio method but needed also here for the method call

        Width = np.abs(LocalX[0] - LocalX[-1])
        Area = np.trapz(Shape, LocalX)

        # alpha = np.array([brentq(
        #     self.findAlpha, 
        #     1e-12,           # Lower bound
        #     np.pi - 1e-12,   # Upper bound  
        #     args=(Area, Width),
        #     xtol=1e-8
        # )])

        root_aplha = root(self.findAlpha, np.pi / 100, args=(Area, Width))
        alpha = root_aplha.x

        RadiusEqv = Width / 2 / np.sin(alpha / 2)
        approach_0 = RadiusEqv * 2 * np.sin(alpha / 4) ** 2
        self.approach = approach_0 / 0.55

        xl = np.sqrt(2 * Rlocal * Shape)
        yl = LocalX

        self.semi_axis_a = np.max(xl)
        self.semi_axis_b = np.abs(yl[-1] - yl[0]) / 2

        self.A_Hertz = self.approach / (self.semi_axis_a) ** 2
        self.B_Hertz = self.approach / (self.semi_axis_b) ** 2

        centroid = 1 / np.trapz(xl, yl) * np.trapz((yl) * xl, yl)

        self.teta_Hertz = (
            np.arccos(
                (self.A_Hertz - self.B_Hertz)
                / (self.A_Hertz + self.B_Hertz)
            )
            * 180
            / np.pi
        )[0]

        self.ratio_Hertz = (
            np.sqrt(1 - (self.teta_Hertz - 90) ** 2 / (90**2))
            ** (EQU_EL_POWER_CORRECTION)
            / EQU_EL_DIV_CORRECTION
        )

        self.normal_force = (
            4
            / 3
            * material_model.Estar
            * np.sqrt(
                1
                / (self.A_Hertz + self.B_Hertz)
                * (self.approach / self.ratio_Hertz) ** 3
            )[0]
        )

        self.Rlocal = Rlocal

        return centroid
    
    def apply_centroid(
        self,
        centroid: float,
        LocalX: np.ndarray[float],
        contact_indexes: np.ndarray[int],
        Wheel: wheel, # kept for future compatibility
        wheel_interp: np.ndarray[float],
        Rail: rail,
        radius_interp: np.ndarray[float], # kept for future compatibility
        WheelCoorLocalNormal: np.ndarray[float],
        RailCoorLocalNormal: np.ndarray[float],
        RotationMatrix: np.ndarray[float, float],
        ContactRotationAngle: float,
    ):
        
        idx0 = contact_indexes[0]

        wheel_y = interpolator(centroid, LocalX, 
                            WheelCoorLocalNormal - wheel_interp[idx0])
        rail_y = interpolator(centroid, LocalX,
                        RailCoorLocalNormal - Rail.rail_profile_pos[idx0, 1])

        centroidWheelNormal = np.array(
            [
                [centroid],
                [wheel_y],
                [0],
            ]
        )
        centroidRailNormal = np.array(
            [
                [centroid],
                [rail_y],
                [0],
            ]
        )

        # TODO: need to add the vertical location of the centraid.
        # At the moment it is a float number but should be a coordinate vector
        centroidWheel_1 = RotationMatrix.T @ centroidWheelNormal
        centroidRail_1 = RotationMatrix.T @ centroidRailNormal

        rail_x_offset = Rail.rail_profile_pos[contact_indexes[0], 0]
        self.centroid_wheel = (centroidWheel_1[0] + rail_x_offset)
        self.centroid_rail = (centroidRail_1[0] + rail_x_offset)

        # TODO:
        # centroidRadius = np.interp(
        #     self.centroid_wheel,
        #     Rail.rail_profile_pos[contact_indexes, 0],
        #     radius_interp[contact_indexes],
        # )

        # WheelAngleCentroid = np.interp(
        #     # centroidWheel, Wheel[:, 0] + SearchPath.DyWheeslet, Wheel.wheel_angle # Wheel[:, 0] + SearchPath.DyWheeslet must be done outside in an object oriented way
        #     self.centroid_rail,
        #     Wheel.wheel_profile_pos[:, 0],
        #     Wheel.wheel_angle,
        # )
        # missing the equivalent WheelAngleCentroid in this version

        self.Q_force = self.normal_force * np.cos(ContactRotationAngle)
        self.Y_force = self.normal_force * np.sin(ContactRotationAngle)
        self.contact_angle = ContactRotationAngle

# ----------------------------------------------------------------- #
# Kik Piotrowski Method                                             #
# ----------------------------------------------------------------- #
class kik_pio_normal_contact(normal_contact):
    def __init__(
        self,
        pressure_0 = 0,
        integral_D1 = 0,
        integral_D3 = 0,
        x_patch = [],
        y_patch = [],
        pressure_patch = [],
        Rlocal = 0,
    ):
        super().__init__()
        self.pressure_0 = pressure_0
        self.integral_D1 = integral_D1
        self.integral_D3 = integral_D3
        self.x_patch = x_patch
        self.y_patch = y_patch
        self.pressure_patch = pressure_patch
        self.Rlocal = Rlocal

    def reset_to_zero(self):
        for attribute in vars(self):
            setattr(self, attribute, 0)

    def calculate_contact(
        self,
        LocalX: np.ndarray[float],
        Shape_kik_pyo: np.ndarray[float],
        Rlocal: np.ndarray[float],
        approach_0: float,
        material_model: material,
        discretization: int,
    ):

        xl = np.sqrt(2 * Rlocal * Shape_kik_pyo)
        yl = LocalX

        # Calculate the centroid
        xl_integral = np.trapz(xl, yl)
        centroid = np.trapz(yl * xl, yl) / xl_integral if xl_integral != 0 else 0

        self.approach = approach_0

        # Create integration grid
        x_integration = np.linspace(-np.max(xl), np.max(xl), discretization)
        y_integration = np.linspace(yl[0] - centroid, yl[-1] - centroid, discretization)
        x_interp = np.interp(y_integration, yl - centroid, xl)
        xl_0 = np.interp(0, y_integration, x_interp)

        # Preallocate arrays for integrals
        search_matrix_1 = np.zeros((discretization, discretization))
        search_matrix_3 = np.zeros((discretization, discretization))

        # Calculate search matrices and integrals
        for jj in range(discretization):
            valid_x = (x_integration >= -x_interp[jj]) & (x_integration <= x_interp[jj])
            search_matrix_1[jj, valid_x] = np.sqrt(
                x_interp[jj] ** 2 - x_integration[valid_x] ** 2
            )
            search_matrix_3[jj, valid_x] = search_matrix_1[jj, valid_x] / np.sqrt(
                (y_integration[jj] ** 2) + x_integration[valid_x] ** 2
            )

        # Compute integral arrays
        integralD1 = np.trapz(search_matrix_1, x_integration, axis=1)
        integralD3 = np.trapz(search_matrix_3, x_integration, axis=1)

        # Aggregate results for D1 and D3
        self.integral_D1 = np.trapz(integralD1, y_integration)
        self.integral_D3 = np.trapz(integralD3, y_integration)

        # Calculate normal force and pressure
        self.normal_force = (
            material_model.kik_pio_constant
            * approach_0
            / 0.55
            * self.integral_D1
            / self.integral_D3
        )
        self.pressure_0 = (
            self.normal_force
            * np.sqrt(2 * Rlocal * approach_0)
            / self.integral_D1
        )

        # Store the pressure distribution output
        self.x_patch = x_integration
        self.y_patch = y_integration
        self.pressure_patch = self.pressure_0 / xl_0 * search_matrix_1

        self.Rlocal = Rlocal
        
        return centroid
    
    def apply_centroid(
        self,
        centroid: float,
        LocalX: np.ndarray[float],
        contact_indexes: np.ndarray[int],
        Wheel: wheel, # kept for future compatibility
        wheel_interp: np.ndarray[float],
        Rail: rail,
        radius_interp: np.ndarray[float], # kept for future compatibility
        WheelCoorLocalNormal: np.ndarray[float],
        RailCoorLocalNormal: np.ndarray[float],
        RotationMatrix: np.ndarray[float, float],
        ContactRotationAngle: float,
    ):
        centroidWheelNormal = np.array(
            [
                [centroid],
                [
                    np.interp(
                        centroid,
                        LocalX,
                        WheelCoorLocalNormal - wheel_interp[contact_indexes[0]],
                    )
                ],
                [0],
            ]
        )
        centroidRailNormal = np.array(
            [
                [centroid],
                [
                    np.interp(
                        centroid,
                        LocalX,
                        RailCoorLocalNormal - Rail.rail_profile_pos[contact_indexes[0], 1],
                    )
                ],
                [0],
            ]
        )

        centroidWheel_1 = RotationMatrix.T @ centroidWheelNormal
        centroidRail_1 = RotationMatrix.T @ centroidRailNormal
        self.centroid_wheel = (
            centroidWheel_1[0] + Rail.rail_profile_pos[contact_indexes[0], 0]
        )
        self.centroid_rail = (
            centroidRail_1[0] + Rail.rail_profile_pos[contact_indexes[0], 0]
        )

        # centroidRadius = np.interp(
        #     self.centroid_wheel,
        #     Rail.rail_profile_pos[contact_indexes, 0],
        #     radius_interp[contact_indexes],
        # )

        # WheelAngleCentroid = np.interp(
        #     # centroidWheel, Wheel[:, 0] + SearchPath.DyWheeslet, Wheel.wheel_angle # Wheel[:, 0] + SearchPath.DyWheeslet must be done outside in an object oriented way
        #     self.centroid_rail,
        #     Wheel.wheel_profile_pos[:, 0],
        #     Wheel.wheel_angle,
        # )
        # missing the equivalent WheelAngleCentroid
        self.Q_force = self.normal_force * np.cos(ContactRotationAngle)
        self.Y_force = self.normal_force * np.sin(ContactRotationAngle)
        self.contact_angle = ContactRotationAngle
