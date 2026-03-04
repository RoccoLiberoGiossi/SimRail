from lib.contact_settings import *
from lib.geoLib.contact_wheel_rail import wheel, rail
from lib.normLib.contact_methods import eqv_el_normal_contact, kik_pio_normal_contact
from lib.contact_material import material
from lib.jittable_functions import interpolator

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

        rail_x = Rail.rail_profile_pos[:, 0]
        rail_y = Rail.rail_profile_pos[:, 1]
        
        WheelInt = interpolator(
            rail_x,
            Wheel.wheel_profile_pos[:, 0],
            Wheel.wheel_profile_pos[:, 1],
        )
        radius = interpolator(
            rail_x,
            Wheel.wheel_radius[:, 0],
            Wheel.wheel_radius[:, 1],
        )

        # ContactPos = np.where((WheelInt - Rail.rail_profile_pos[:, 1]) > 0)[0]
        contact_mask = WheelInt > rail_y
        ContactPos = np.flatnonzero(contact_mask)
        
        if len(ContactPos) <= 2:
            for patch in self.contact_patches[:self.n_patches]:
                patch.reset_to_zero()
            return
        
        gaps = np.diff(ContactPos) > 1

        if not np.any(gaps):
            # Single continuous region
            ContactPosNs = [ContactPos]
        else:
            # Split at gaps
            split_indices = np.flatnonzero(gaps) + 1
            ContactPosNs = np.split(ContactPos, split_indices)

        if len(ContactPosNs) > 1:
            ContactPosNs_sorted = sorted(ContactPosNs, key=len, reverse=True)
        else:
            ContactPosNs_sorted = ContactPosNs

        for patch in self.contact_patches[:self.n_patches]:
            patch.reset_to_zero()

        n_active = min(self.n_patches, len(ContactPosNs_sorted))
    
        for counter in range(n_active):
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

        n_contacts = len(contact_indexes)
        rail_x = Rail.rail_profile_pos[contact_indexes, 0]
        rail_y = Rail.rail_profile_pos[contact_indexes, 1]

        WheelCoor = np.empty((3, n_contacts))
        WheelCoor[0, :] = rail_x - rail_x[0]
        WheelCoor[1, :] = wheel_interp[contact_indexes] - wheel_interp[contact_indexes[0]]
        WheelCoor[2, :] = 0
        
        RadiusCoor = np.empty((3, n_contacts))
        RadiusCoor[0, :] = rail_x - rail_x[0]
        RadiusCoor[1, :] = radius_interp[contact_indexes] - radius_interp[contact_indexes[0]]
        RadiusCoor[2, :] = 0
        
        RailCoor = np.empty((3, n_contacts))
        RailCoor[0, :] = rail_x - rail_x[0]
        RailCoor[1, :] = rail_y - rail_y[0]
        RailCoor[2, :] = 0

        # TODO: in future, the rotation matrix must be substituted with quaternion for a better
        # representation and a more stable computation.
        cos_alpha = np.cos(ContactRotationAngle)
        sin_alpha = np.sin(ContactRotationAngle)

        RotationMatrix = np.array(
            [
                [cos_alpha, -sin_alpha, 0],
                [sin_alpha, cos_alpha, 0],
                [0, 0, 1],
            ]
        )

        WheelCoorLocal = RotationMatrix @ WheelCoor
        RadiusCoorLocal = RotationMatrix @ RadiusCoor
        RailCoorLocal = RotationMatrix @ RailCoor

        LocalX = np.linspace(RailCoorLocal[0, 0], RailCoorLocal[0, -1], num=discretization)

        WheelCoorLocalNormal = (
            interpolator(LocalX, WheelCoorLocal[0, :], WheelCoorLocal[1, :])
            + wheel_interp[contact_indexes[0]]
        )
        RadiusCoorLocalNormal = (
            interpolator(LocalX, RadiusCoorLocal[0, :], RadiusCoorLocal[1, :])
            + radius_interp[contact_indexes[0]]
        )
        RailCoorLocalNormal = (
            interpolator(LocalX, RailCoorLocal[0, :], RailCoorLocal[1, :])
            + Rail.rail_profile_pos[contact_indexes[0], 1]
        )

        if isinstance(self.contact_model, eqv_el_normal_contact):
            # assign radius for eq contact model
            Rlocal = RadiusCoorLocalNormal[int(discretization / 2)] / cos_alpha
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

            Rlocal = RadiusCoorLocalNormal / cos_alpha

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
