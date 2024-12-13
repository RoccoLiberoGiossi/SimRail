class normal_contact:
    def __init__(self, approach=0, normal_force=0, centroid=0, Q_force=0, Y_force=0):
        self.approach = approach
        self.normal_force = normal_force
        self.Q_force = Q_force
        self.Y_force = Y_force
        self.centroid = centroid

    def reset_to_zero(self):
        for attribute in vars(self):
            setattr(self, attribute, 0)


class eqv_el_normal_contact(normal_contact):
    def __init__(
        self, semi_axis_a, semi_axis_b, A_Hertz, B_Hertz, teta_Hertz, ratio_Hertz
    ):
        super().__init__()
        self.semi_axis_a = semi_axis_a
        self.semi_axis_b = semi_axis_b
        self.A_Hertz = A_Hertz
        self.B_Hertz = B_Hertz
        self.teta_Hertz = teta_Hertz
        self.ratio_Hertz = ratio_Hertz

    def reset_to_zero(self):
        for attribute in vars(self):
            setattr(self, attribute, 0)
