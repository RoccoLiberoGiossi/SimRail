from lib.contact_settings import *
from lib.geoLib.contact_wheel_rail import wheel, rail
from lib.normLib.contact_methods import eqv_el_normal_contact, kik_pio_normal_contact
from lib.contact_material import material

class NormalContactSolver:
    def __init__(
        self,
        contact_model: Union[eqv_el_normal_contact, kik_pio_normal_contact],
        numer_of_patches: int = 3,
    ):

        if not isinstance(numer_of_patches, int):
            raise TypeError(
                "Error in contact forces calculation: numer_of_patches is not an integer"
            )
        
        if not isinstance(contact_model, (eqv_el_normal_contact, kik_pio_normal_contact)):
            raise TypeError("Error in contact forces calculation: no contact model provided")
        
        self.contact_model = contact_model
        
        if isinstance(self.contact_model, eqv_el_normal_contact):
            self.n_patches = numer_of_patches
            self.contact_patches = [type(self.contact_model)() for _ in range(self.n_patches)]
        elif isinstance(self.contact_model, kik_pio_normal_contact):
            self.n_patches = numer_of_patches * 3
            self.contact_patches = [type(self.contact_model)() for _ in range(self.n_patches)]

    def patch_search(
        self,
        Wheel: wheel,
        Rail: rail,
        material_model: material,
        discretization: int = 58
    ):

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

            for counter in range(self.n_patches):
                self.contact_patches[counter].reset_to_zero()

            for counter in range(self.n_patches):
                if counter < len(ContactPosNs_sorted):
                    self.contact_forces(
                        ContactPosNs_sorted[counter],
                        Wheel,
                        WheelInt,
                        Rail,
                        radius,
                        discretization,
                        counter,
                        material_model,
                    )
        else:
            for counter in range(self.n_patches):
                self.contact_patches[counter].reset_to_zero()

    def contact_forces(
        self,
        contact_indexes: np.ndarray[int],
        Wheel: wheel,
        wheel_interp: np.ndarray[float],
        Rail: rail,
        radius_interp: np.ndarray[float],
        discretization: int,
        counter: int,
        material_model: material,
    ):
        """contact_indexes = indexes of wheel and rail where interpenetration is found
        Wheel = wheel object
        wheel_interp = wheel segment interpolated over the rail
        Rail = rail object
        radius_interp = wheel radius segment interpolated over the rail
        discretization = patch discretization as imposed by engineer"""

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

        # TODO: in future, the rotation matrix must be substituted with quaternion for a better
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

        if isinstance(self.contact_model, eqv_el_normal_contact):
            # assign radius for eq contact model
            Rlocal = RadiusCoorLocalNormal[int(discretization / 2)] / np.cos(
                ContactRotationAngle
            )
            # assign shape of the contact patch for the equivalent contact model
            Shape = WheelCoorLocalNormal - RailCoorLocalNormal
            # assign contact model for the local contact patch
            contact_model = self.contact_patches[counter]
            # calculate centroid
            centroid = contact_model.calculate_contact(
                LocalX,
                Shape,
                Rlocal,
                material_model,
            )
            contact_model.apply_centroid(
                centroid,
                LocalX,
                contact_indexes,
                Wheel,
                wheel_interp,
                Rail,
                radius_interp,
                WheelCoorLocalNormal,
                RailCoorLocalNormal,
                RotationMatrix,
                ContactRotationAngle,
            )

        elif isinstance(self.contact_model, kik_pio_normal_contact):

            Rlocal = RadiusCoorLocalNormal / np.cos(
                ContactRotationAngle
            )

            max_shape = np.max(WheelCoorLocalNormal - RailCoorLocalNormal)
            approach_0 = 0.55 * max_shape
            approach_1 = 0.45 * max_shape

            # Adjusting Shape for the Kik-Piotrowski method
            WheelCoorLocalNormal_kik_pyo = WheelCoorLocalNormal - approach_1
            Shape = WheelCoorLocalNormal_kik_pyo - RailCoorLocalNormal

            # find indexes for new shape greter then 0
            Contact_kik_pio = np.where((Shape) > 0)[0]

            # split indexes to create multiple patches
            indices = np.where(np.diff(Contact_kik_pio) > 1)[0] + 1
            Contact_kik_pio_Ns = np.split(Contact_kik_pio, indices)

            # cicle over the splitted shapes
            for counter_kik_pio in range(len(Contact_kik_pio_Ns)):
                # define the contact model for the patch (double for kik)
                contact_model = self.contact_patches[counter+(self.n_patches//3)*counter_kik_pio]

                LocalX_kik_pio = LocalX[Contact_kik_pio_Ns[counter_kik_pio]]
                Shape_kik_pio = Shape[Contact_kik_pio_Ns[counter_kik_pio]]
                approach_kik_pio = np.max(Shape_kik_pio)
                Rlocal_kik_pio = Rlocal[Contact_kik_pio_Ns[counter_kik_pio]]

                centroid = contact_model.calculate_contact(
                    LocalX_kik_pio,
                    Shape_kik_pio,
                    Rlocal_kik_pio[len(Rlocal_kik_pio)//2],
                    approach_kik_pio,
                    material_model,
                    discretization,
                )

                contact_model.apply_centroid(
                    centroid,
                    LocalX_kik_pio,
                    contact_indexes,
                    Wheel,
                    wheel_interp,
                    Rail,
                    radius_interp,
                    WheelCoorLocalNormal[Contact_kik_pio_Ns[counter_kik_pio]],
                    RailCoorLocalNormal[Contact_kik_pio_Ns[counter_kik_pio]],
                    RotationMatrix, # this should be adjusted when a second contact occours
                    ContactRotationAngle, # this should be adjusted when a second contact occours
                )
