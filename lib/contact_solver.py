from lib.contact_settings import *

def findAlpha(alpha, Area, Width):
    Res = Area-0.5*Width**2/4*(alpha-np.sin(alpha))/(np.sin(alpha/2)**2)
    return Res

def PatchSearch(
    contact_indexes, Wheel, wheel_interp, Rail, radius, discretization
):
    """ contact_indexes = indexes of wheel and rail where interpenetration is found 
        Wheel = wheel profile at current state
        wheel_interp = wheel segment interpolated over the rail
        Rail = rail profile at current state
        radius = wheel radius at current state
        discretization = patch discretization as imposed by engineer """

    ContactRotationAngle = -(
        np.arctan(
            (Rail[contact_indexes[-1], 1] - Rail[contact_indexes[0], 1])
            / (Rail[contact_indexes[-1], 0] - Rail[contact_indexes[0], 0])
        )
    )

    WheelCoor = np.vstack(
        (
            np.reshape(Rail[contact_indexes, 0], [1, len(contact_indexes)])
            - Rail[contact_indexes[0], 0],
            np.reshape(wheel_interp[contact_indexes], [1, len(contact_indexes)])
            - wheel_interp[contact_indexes[0]],
            np.zeros([1, len(contact_indexes)]),
        )
    )
    RadiusCoor = np.vstack(
        (
            np.reshape(Rail[contact_indexes, 0], [1, len(contact_indexes)])
            - Rail[contact_indexes[0], 0],
            np.reshape(radius[contact_indexes], [1, len(contact_indexes)])
            - radius[contact_indexes[0]],
            np.zeros([1, len(contact_indexes)]),
        )
    )
    RailCoor = np.vstack(
        (
            np.reshape(Rail[contact_indexes, 0], [1, len(contact_indexes)])
            - Rail[contact_indexes[0], 0],
            np.reshape(Rail[contact_indexes, 1], [1, len(contact_indexes)])
            - Rail[contact_indexes[0], 1],
            np.zeros([1, len(contact_indexes)]),
        )
    )

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
        + radius[contact_indexes[0]]
    )
    RailCoorLocalNormal = (
        np.interp(LocalX, RailCoorLocal[0, :], RailCoorLocal[1, :])
        + Rail[contact_indexes[0], 1]
    )

    Shape = WheelCoorLocalNormal - RailCoorLocalNormal
    Width = np.abs(LocalX[-1] - LocalX[0])
    Area = np.trapz(Shape, LocalX)
    root = fsolve(findAlpha, np.pi / 100, args=(Area, Width))
    alpha = root
    RadiusEqv = Width / 2 / np.sin(alpha / 2)
    approach_0 = RadiusEqv * 2 * np.sin(alpha / 4) ** 2
    approach = approach_0 / 0.55

    Rlocal = RadiusCoorLocalNormal[int(discretization / 2)] / np.cos(
        ContactRotationAngle
    )
    xl = np.sqrt(2 * Rlocal * Shape)
    yl = LocalX

    aHertzCorrected = np.max(xl)
    bHertzCorrected = np.abs(yl[-1] - yl[0]) / 2

    AHertz = approach / (aHertzCorrected) ** 2
    BHertz = approach / (bHertzCorrected) ** 2

    centroid = 1 / np.trapz(xl, yl) * np.trapz((yl) * xl, yl)

    centroidWheelNormal = np.array(
        [
            [centroid],
            [
                np.interp(
                    centroid, LocalX, WheelCoorLocalNormal - wheel_interp[contact_indexes[0]]
                )
            ],
            [0],
        ]
    )
    centroidRailNormal = np.array(
        [
            [centroid],
            [np.interp(centroid, LocalX, RailCoorLocalNormal - Rail[contact_indexes[0], 1])],
            [0],
        ]
    )

    centroidWheel_1 = np.matmul(np.linalg.inv(RotationMatrix), centroidWheelNormal)
    centroidRail_1 = np.matmul(np.linalg.inv(RotationMatrix), centroidRailNormal)
    centroidWheel = centroidWheel_1[0] + Rail[contact_indexes[0], 0]
    centroidRail = centroidRail_1[0] + Rail[contact_indexes[0], 0]

    centroidRadius = np.interp(centroidWheel, Rail[contact_indexes, 0], radius[contact_indexes])

    TetaHertz = np.arccos((AHertz - BHertz) / (AHertz + BHertz)) * 180 / np.pi

    coeff_power = 1.026600611713163413e00
    coeff_div = 1.397839912270349316e00

    ratioHertz = (
        np.sqrt(1 - (TetaHertz - 90) ** 2 / (90**2)) ** (coeff_power) / coeff_div
    )

    NormalForceHertz = (
        4 / 3 * Estar * np.sqrt(1 / (AHertz + BHertz) * (approach / ratioHertz) ** 3)
    )

    WheelAngleCentroid = np.interp(
        centroidWheel, Wheel[:, 0] + SearchPath.DyWheeslet, WheelAngle
    )
    QForce = NormalForceHertz * np.cos(ContactRotationAngle)

    SearchPath.WheelAngle = WheelAngleCentroid
    SearchPath.approach = approach
    SearchPath.NormalForce = NormalForceHertz
    SearchPath.QForce = QForce
    SearchPath.Deltaz = deltaz
    SearchPath.centroid = centroidWheel - Search1.DyWheeslet
    SearchPath.DyWheel = centroidWheel - Search1.DyWheeslet
    SearchPath.a = aHertzCorrected
    SearchPath.b = bHertzCorrected
    SearchPath.AHertz = AHertz
    SearchPath.BHertz = BHertz
    SearchPath.rWy = WheelCoorLocalNormal
    SearchPath.rWx = RailCoorLocalNormal
    SearchPath.rRy = LocalX
    SearchPath.tetaHertz = TetaHertz
    SearchPath.ratioHertz = ratioHertz
    SearchPath.DyRail = centroidRail
    SearchPath.StartPos = yl[0] + Rail[contact_indexes[0], 0] - Search1.DyWheeslet
    SearchPath.EndPos = yl[-1] + Rail[contact_indexes[0], 0] - Search1.DyWheeslet
    SearchPath.Radius = centroidRadius  # Rlocal*np.cos(ContactRotationAngle)

    return QForce
