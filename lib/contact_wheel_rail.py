from lib.contact_settings import *
from lib.contact_dyn_state import dynamic_state

# --------------------------------------------------------- #
# Define wheel and rail parameters                          #
# this function interpolates the points to a "standard"     #
# discratization. It can be improve.                        #
# --------------------------------------------------------- #


def calculate_profile(start_profile):

    DeltaX = 0.025

    num_profile = np.arange(0, len(start_profile) - DeltaX, DeltaX)
    num_profile_old = np.arange(len(start_profile))
    y_profile = interpolate.PchipInterpolator(num_profile_old, start_profile[:, 0])(
        num_profile
    )
    z_profile = interpolate.PchipInterpolator(start_profile[:, 0], start_profile[:, 1])(
        y_profile
    )

    output_profile = np.zeros([len(z_profile), 2])
    output_profile[:, 0] = y_profile
    output_profile[:, 1] = -z_profile

    return output_profile


class wheel:
    def __init__(self, wheel_profile=S1002_profile, b0=0.75, r0=0.46, lr=RIGHT):
        self.wheel_profile = calculate_profile(wheel_profile)
        self.b0 = b0
        self.r0 = r0
        self.lr = lr
        self.dynamic_state = dynamic_state()
        self.wheel_start_pos = self.set_start_post()
        self.wheel_profile_pos = self.wheel_start_pos
        self.wheel_start_radius = self.calc_radius()
        self.wheel_start_angle = self.calc_wheel_angle()
        self.wheel_radius = self.wheel_start_radius
        self.wheel_angle = self.wheel_start_angle

    def calc_radius(self):
        Indexes = np.where(self.wheel_profile[:, 0] >= 0)
        wheel_radius = np.zeros([len(self.wheel_profile), 2])
        wheel_radius[:, 0] = self.wheel_profile[:, 0]
        wheel_radius[:, 1] = (
            self.r0 + self.wheel_profile[:, 1] - self.wheel_profile[Indexes[0][0], 1]
        )
        if self.lr == RIGHT:
            wheel_radius[:, 0] = wheel_radius[:, 0] + self.b0
            wheel_radius[:, 1] = wheel_radius[:, 1]
        elif self.lr == LEFT:
            wheel_radius[:, 0] = -np.flip(wheel_radius[:, 0]) - self.b0
            wheel_radius[:, 1] = np.flip(wheel_radius[:, 1])
        else:
            print("error in wheel location")

        return wheel_radius

    def set_start_post(self):
        wheel_start_pos = np.zeros([len(self.wheel_profile), 2])
        if self.lr == RIGHT:
            wheel_start_pos[:, 0] = self.wheel_profile[:, 0] + self.b0
            wheel_start_pos[:, 1] = self.wheel_profile[:, 1]
        elif self.lr == LEFT:
            wheel_start_pos[:, 0] = -np.flip(self.wheel_profile[:, 0]) - self.b0
            wheel_start_pos[:, 1] = np.flip(self.wheel_profile[:, 1])
        else:
            print("Please add if it is the left or right wheel")

        return wheel_start_pos

    def calc_wheel_angle(self):
        if self.lr == RIGHT:
            wheel_angle = np.arctan(
                (self.wheel_profile_pos[0:-1, 1] - self.wheel_profile_pos[1:, 1])
                / (self.wheel_profile_pos[0:-1, 0] - self.wheel_profile_pos[1:, 0])
            )
            wheel_angle = np.append(wheel_angle, wheel_angle[-1])
        elif self.lr == LEFT:
            wheel_angle = -np.arctan(
                (self.wheel_profile_pos[1:, 1] - self.wheel_profile_pos[0:-1, 1])
                / (self.wheel_profile_pos[0:-1, 0] - self.wheel_profile_pos[1:, 0])
            )
            wheel_angle = np.append(wheel_angle[0], wheel_angle)
        else:
            print("Please add if it is the left or right wheel")

        return wheel_angle

    def set_dynamic_state(self, new_state):
        if isinstance(new_state, dynamic_state):
            self.dynamic_state = new_state
        else:
            raise TypeError("new_state must be an instance of dynamic_state")

    def calculate_position(self):
        temp_profile = self.calculate_yaw()
        temp_profile[:, 0] = temp_profile[:, 0] + self.dynamic_state.state_y
        temp_profile[:, 1] = temp_profile[:, 1] + self.dynamic_state.state_z

        self.wheel_profile_pos = temp_profile
        self.wheel_angle = self.calc_wheel_angle()

    def calculate_yaw(self):
        pos_yawed = np.zeros([len(self.wheel_start_pos), 2])
        pos_yawed[:, 0] = (self.wheel_start_pos[:, 0]) * np.cos(
            self.dynamic_state.state_yaw
        ) - self.wheel_start_radius[:, 1] * np.tan(self.wheel_start_angle) * (
            np.sin(self.dynamic_state.state_yaw)
        ) ** 2 / np.cos(
            self.dynamic_state.state_yaw
        )
        rotation_P = (
            1
            - (1 + (np.tan(self.wheel_start_angle)) ** 2)
            * (np.sin(self.dynamic_state.state_yaw)) ** 2
        )
        pos_yawed[:, 1] = (
            self.wheel_start_radius[:, 1]
            * np.sqrt(rotation_P)
            / np.cos(self.dynamic_state.state_yaw)
            - self.r0
        )

        return pos_yawed

    def calculate_roll():
        # TO DO
        pass


class rail:
    def __init__(
        self, rail_profile=UIC60_profile, gauge=1.435, rail_inclination=40, lr=RIGHT
    ):
        self.rail_profile = calculate_profile(rail_profile)
        self.gauge = gauge
        self.lr = lr
        self.rail_inclination = rail_inclination
        self.rail_cant = -np.arctan(1 / self.rail_inclination)
        self.rail_rotated = self.rotate_profile()
        self.gauge_corner = np.where(self.rail_rotated[:, 1] <= 0.014)[0]
        self.rail_start_pos = self.set_start_post()

    def rotate_profile(self):
        rotation_Z = np.array(
            [
                [np.cos(self.rail_cant), -np.sin(self.rail_cant), 0],
                [np.sin(self.rail_cant), np.cos(self.rail_cant), 0],
                [0, 0, 1],
            ]
        )

        rail_rotated = np.zeros([len(self.rail_profile), 2])

        for jj in range(0, len(self.rail_profile)):
            Coord = np.array([self.rail_profile[jj, 0], self.rail_profile[jj, 1], 0])
            RailTiltLocal = np.matmul(rotation_Z, np.transpose(Coord))
            rail_rotated[jj, 0] = RailTiltLocal[0]
            rail_rotated[jj, 1] = RailTiltLocal[1]

        return rail_rotated

    def set_start_post(self):
        rail_start_pos = np.zeros([len(self.rail_rotated), 2])
        if self.lr == RIGHT:
            rail_start_pos[:, 0] = (
                self.rail_rotated[:, 0]
                - self.rail_rotated[self.gauge_corner[0], 0]
                + self.gauge / 2
            )
            rail_start_pos[:, 1] = self.rail_rotated[:, 1]
        elif self.lr == LEFT:
            rail_start_pos[:, 0] = (
                -np.flip(self.rail_rotated[:, 0])
                + self.rail_rotated[self.gauge_corner[0], 0]
                - self.gauge / 2
            )
            rail_start_pos[:, 1] = np.flip(self.rail_rotated[:, 1])
        else:
            print("Please add if it is the left or right rail")

        return rail_start_pos
