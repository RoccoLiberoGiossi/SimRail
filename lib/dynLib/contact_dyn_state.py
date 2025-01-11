class dynamic_state:
    def __init__(
        self,
        state_y=0,
        state_z=0,
        state_yaw=0,
        state_roll=0,
        state_v_y=0,
        state_v_z=0,
        state_v_yaw=0,
        state_v_roll=0,
        state_a_y=0,
        state_a_z=0,
        state_a_yaw=0,
        state_a_roll=0,
    ):
        self.state_y = state_y
        self.state_z = state_z
        self.state_yaw = state_yaw
        self.state_roll = state_roll
        self.state_v_y = state_v_y
        self.state_v_z = state_v_z
        self.state_v_yaw = state_v_yaw
        self.state_v_roll = state_v_roll
        self.state_a_y = state_a_y
        self.state_a_z = state_a_z
        self.state_a_yaw = state_a_yaw
        self.state_a_roll = state_a_roll
