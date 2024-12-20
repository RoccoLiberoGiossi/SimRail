from lib.contact_settings import *
from lib.contact_wheel_rail import wheel, rail
from lib.contact_methods import eqv_el_normal_contact
from lib.contact_material import material


def findAlpha(alpha, Area, Width):
    Res = Area - 0.5 * Width**2 / 4 * (alpha - np.sin(alpha)) / (np.sin(alpha / 2) ** 2)
    return Res


def calculate_contact_eqv(contact_model, LocalX, Shape, Rlocal, material_model):

    Width = np.abs(LocalX[0] - LocalX[-1])
    Area = np.trapz(Shape, LocalX)
    root = fsolve(findAlpha, np.pi / 100, args=(Area, Width))
    alpha = root
    RadiusEqv = Width / 2 / np.sin(alpha / 2)
    approach_0 = RadiusEqv * 2 * np.sin(alpha / 4) ** 2
    contact_model.approach = approach_0 / 0.55

    xl = np.sqrt(2 * Rlocal * Shape)
    yl = LocalX

    contact_model.semi_axis_a = np.max(xl)
    contact_model.semi_axis_b = np.abs(yl[-1] - yl[0]) / 2

    contact_model.A_Hertz = contact_model.approach / (contact_model.semi_axis_a) ** 2
    contact_model.B_Hertz = contact_model.approach / (contact_model.semi_axis_b) ** 2

    centroid = 1 / np.trapz(xl, yl) * np.trapz((yl) * xl, yl)

    contact_model.teta_Hertz = (
        np.arccos(
            (contact_model.A_Hertz - contact_model.B_Hertz)
            / (contact_model.A_Hertz + contact_model.B_Hertz)
        )
        * 180
        / np.pi
    )

    contact_model.ratio_Hertz = (
        np.sqrt(1 - (contact_model.teta_Hertz - 90) ** 2 / (90**2))
        ** (EQU_EL_POWER_CORRECTION)
        / EQU_EL_DIV_CORRECTION
    )

    contact_model.normal_force = (
        4
        / 3
        * material_model.Estar  # I don't like it like this since it has to go around
        * np.sqrt(
            1
            / (contact_model.A_Hertz + contact_model.B_Hertz)
            * (contact_model.approach / contact_model.ratio_Hertz) ** 3
        )
    )

    return centroid


def contact_forces(
    contact_indexes,
    Wheel,
    wheel_interp,
    Rail,
    radius_interp,
    discretization,
    contact_model,
    material_model,
):
    """contact_indexes = indexes of wheel and rail where interpenetration is found
    Wheel = wheel object
    wheel_interp = wheel segment interpolated over the rail
    Rail = rail object
    radius_interp = wheel radius segment interpolated over the rail
    discretization = patch discretization as imposed by engineer"""

    if not isinstance(Wheel, wheel):
        raise TypeError("Error in contact forces calculation: wheel not given")

    if not isinstance(Rail, rail):
        raise TypeError("Error in contact forces calculation: rail not given")

    if not isinstance(material_model, material):
        raise TypeError("Error in contact forces calculation: no material provided")

    if not isinstance(discretization, int):
        raise TypeError(
            "Error in contact forces calculation: discretization is not an integer"
        )

    ContactRotationAngle = -(
        np.arctan(
            (
                Rail.rail_profile_pos[contact_indexes[-1], 1]
                - Rail.rail_profile_pos[contact_indexes[0], 1]
            )
            / (
                Rail.rail_profile_pos[contact_indexes[-1], 0]
                - Rail.rail_profile_pos[contact_indexes[0], 0]
            )
        )
    )

    WheelCoor = np.vstack(
        (
            np.reshape(
                Rail.rail_profile_pos[contact_indexes, 0], [1, len(contact_indexes)]
            )
            - Rail.rail_profile_pos[contact_indexes[0], 0],
            np.reshape(wheel_interp[contact_indexes], [1, len(contact_indexes)])
            - wheel_interp[contact_indexes[0]],
            np.zeros([1, len(contact_indexes)]),
        )
    )
    RadiusCoor = np.vstack(
        (
            np.reshape(
                Rail.rail_profile_pos[contact_indexes, 0], [1, len(contact_indexes)]
            )
            - Rail.rail_profile_pos[contact_indexes[0], 0],
            np.reshape(radius_interp[contact_indexes], [1, len(contact_indexes)])
            - radius_interp[contact_indexes[0]],
            np.zeros([1, len(contact_indexes)]),
        )
    )
    RailCoor = np.vstack(
        (
            np.reshape(
                Rail.rail_profile_pos[contact_indexes, 0], [1, len(contact_indexes)]
            )
            - Rail.rail_profile_pos[contact_indexes[0], 0],
            np.reshape(
                Rail.rail_profile_pos[contact_indexes, 1], [1, len(contact_indexes)]
            )
            - Rail.rail_profile_pos[contact_indexes[0], 1],
            np.zeros([1, len(contact_indexes)]),
        )
    )

    # in future, the rotation matrix must bu substituted with aquaternion for a better
    # representation and a more stable computation.
    RotationMatrix = np.array(
        [
            [np.cos(ContactRotationAngle), -np.sin(ContactRotationAngle), 0],
            [np.sin(ContactRotationAngle), np.cos(ContactRotationAngle), 0],
            [0, 0, 1],
        ]
    )

    WheelCoorLocal = RotationMatrix @ WheelCoor
    RadiusCoorLocal = RotationMatrix @ RadiusCoor
    RailCoorLocal = RotationMatrix @ RailCoor

    LocalX = np.linspace(RailCoorLocal[0, 0], RailCoorLocal[0, -1], num=discretization)

    WheelCoorLocalNormal = (
        np.interp(LocalX, WheelCoorLocal[0, :], WheelCoorLocal[1, :])
        + wheel_interp[contact_indexes[0]]
    )
    RadiusCoorLocalNormal = (
        np.interp(LocalX, RadiusCoorLocal[0, :], RadiusCoorLocal[1, :])
        + radius_interp[contact_indexes[0]]
    )
    RailCoorLocalNormal = (
        np.interp(LocalX, RailCoorLocal[0, :], RailCoorLocal[1, :])
        + Rail.rail_profile_pos[contact_indexes[0], 1]
    )

    Rlocal = RadiusCoorLocalNormal[int(discretization / 2)] / np.cos(
        ContactRotationAngle
    )

    Shape = WheelCoorLocalNormal - RailCoorLocalNormal

    if isinstance(contact_model, eqv_el_normal_contact):
        centroid = calculate_contact_eqv(
            contact_model, LocalX, Shape, Rlocal, material_model
        )
    else:
        raise TypeError(
            "Error in contact forces calculation: no contact model provided"
        )

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
                    centroid, LocalX, RailCoorLocalNormal - Rail.rail_profile_pos[contact_indexes[0], 1]
                )
            ],
            [0],
        ]
    )

    centroidWheel_1 = RotationMatrix.T @ centroidWheelNormal
    centroidRail_1 = RotationMatrix.T @ centroidRailNormal
    contact_model.centroid_wheel = (
        centroidWheel_1[0] + Rail.rail_profile_pos[contact_indexes[0], 0]
    )
    contact_model.centroid_rail = (
        centroidRail_1[0] + Rail.rail_profile_pos[contact_indexes[0], 0]
    )

    centroidRadius = np.interp(
        contact_model.centroid_wheel,
        Rail.rail_profile_pos[contact_indexes, 0],
        radius_interp[contact_indexes],
    )

    WheelAngleCentroid = np.interp(
        # centroidWheel, Wheel[:, 0] + SearchPath.DyWheeslet, Wheel.wheel_angle # Wheel[:, 0] + SearchPath.DyWheeslet must be done outside in an object oriented way
        contact_model.centroid_rail,
        Wheel.wheel_profile_pos[:, 0],
        Wheel.wheel_angle,
    )
    # missing the equivalent WheelAngleCentroid
    contact_model.Q_force = contact_model.normal_force * np.cos(ContactRotationAngle)
    contact_model.Y_force = contact_model.normal_force * np.cos(ContactRotationAngle)

    # Keep it here for momentarly back_up
    # SearchPath.WheelAngle = WheelAngleCentroid
    # SearchPath.approach = approach
    # SearchPath.NormalForce = NormalForceHertz
    # SearchPath.QForce = QForce
    # SearchPath.Deltaz = deltaz
    # SearchPath.centroid = centroidWheel - Search1.DyWheeslet
    # SearchPath.DyWheel = centroidWheel - Search1.DyWheeslet
    # SearchPath.a = aHertzCorrected
    # SearchPath.b = bHertzCorrected
    # SearchPath.AHertz = AHertz
    # SearchPath.BHertz = BHertz
    # SearchPath.rWy = WheelCoorLocalNormal
    # SearchPath.rWx = RailCoorLocalNormal
    # SearchPath.rRy = LocalX
    # SearchPath.tetaHertz = TetaHertz
    # SearchPath.ratioHertz = ratioHertz
    # SearchPath.DyRail = centroidRail
    # SearchPath.StartPos = yl[0] + Rail[contact_indexes[0], 0] - Search1.DyWheeslet
    # SearchPath.EndPos = yl[-1] + Rail[contact_indexes[0], 0] - Search1.DyWheeslet
    # SearchPath.Radius = centroidRadius  # Rlocal*np.cos(ContactRotationAngle)

    # return QForce


# deltaz -> need to put this outside togeter with dy of the search. They must be incorporated in the update state
def patch_search(Wheel, Rail, material_model, contact_patches, discretization=58):

    if not isinstance(Wheel, wheel):
        raise TypeError("Error in contact forces calculation: wheel not given")

    if not isinstance(Rail, rail):
        raise TypeError("Error in contact forces calculation: rail not given")

    if not isinstance(material_model, material):
        raise TypeError("Error in contact forces calculation: no material provided")

    WheelInt = np.interp(
        Rail.rail_profile_pos[:, 0],
        Wheel.wheel_profile_pos[:, 0],
        Wheel.wheel_profile_pos[:, 1],
    )
    radius = np.interp(
        Rail.rail_profile_pos[:, 0],
        Wheel.wheel_radius[:, 0],
        Wheel.wheel_radius[:, 1],
    )

    ContactPos = np.where((WheelInt - Rail.rail_profile_pos[:, 1]) > 0)[0]

    if len(ContactPos) > 2:
        
        indices = np.where(np.diff(ContactPos) > 1)[0] + 1
        ContactPosNs = np.split(ContactPos, indices)
        ContactPosNs_sorted = sorted(ContactPosNs, key=len, reverse=True)

        for counter in range(len(contact_patches)):
            if counter < len(ContactPosNs_sorted):
                contact_forces(
                    ContactPosNs_sorted[counter],
                    Wheel,
                    WheelInt,
                    Rail,
                    radius,
                    discretization,
                    contact_patches[counter],
                    material_model,
                )
            else:
                contact_patches[counter].reset_to_zero()

    else:
        for counter in range(len(contact_patches)):
            contact_patches[counter].reset_to_zero()