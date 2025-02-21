from lib.contact_settings import *
from lib.normLib.contact_methods import eqv_el_normal_contact
from lib.contact_material import material
from lib.dynLib.contact_dyn_state import dynamic_state


class tangent_creepage:
    def __init__(self, nu_x=0, nu_y=0, phi=0):
        self.nu_x = nu_x
        self.nu_y = nu_y
        self.phi = phi


class fastsim:
    def __init__(self, discratization=58):
        self.discratization = discratization
        self.F_x = 0
        self.F_y = 0
        self.xx = np.zeros(1)
        self.yy = np.zeros(1)
        self.p_x = np.zeros(1)
        self.p_y = np.zeros(1)
        self.s_x = np.zeros(1)
        self.s_y = np.zeros(1)
        self.s_out = np.zeros(1)
        self.g_bound = np.zeros(1)

    def reset_to_zero(self):
        for attribute in vars(self):
            setattr(self, attribute, 0)

def calc_tangent_force(fastsim_patch, contact_patch, material_properties, slip_state, state_dyn):

    if isinstance(contact_patch, eqv_el_normal_contact):

        if not isinstance(fastsim_patch, fastsim):
            raise TypeError("Error in FASTSIM calculation: not a fastsim method provided")
        if not isinstance(material_properties, material):
            raise TypeError("Error in FASTSIM calculation: not a material method")
        if not isinstance(state_dyn, dynamic_state):
            raise TypeError(
                "Error in FASTSIM calculation: not a correct dynamic state method"
            )
        if not isinstance(slip_state, tangent_creepage):
            raise TypeError(
                "Error in FASTSIM calculation: not a correct dynamic state method"
            )

        a = contact_patch.semi_axis_a
        b = contact_patch.semi_axis_b

        pchip_interp_c_11 = PchipInterpolator(abratio, c11tab)
        pchip_interp_c_22 = PchipInterpolator(abratio, c22tab)
        pchip_interp_c_23 = PchipInterpolator(abratio, c23tab)

        # Interpolating values
        c_11 = pchip_interp_c_11(a / b)
        c_22 = pchip_interp_c_22(a / b)
        c_23 = pchip_interp_c_23(a / b)
        # # Interpolating values
        # c_11 = np.interp(a / b, abratio, c11tab)
        # c_22 = np.interp(a / b, abratio, c22tab)
        # c_23 = np.interp(a / b, abratio, c23tab)

        # Calculating L_x, L_y, and L_phi
        L_x = 8 * a / (3 * material_properties.G * c_11)
        L_y = 8 * a / (3 * material_properties.G * c_22)
        L_phi = np.pi * a**2 / (4 * material_properties.G * np.sqrt(a * b) * c_23)

        # Calculating r and q
        r = 2 * b / (fastsim_patch.discratization - 1)  # d_b
        yy = np.arange(-b, b + r, r)
        q = 2 * a / (fastsim_patch.discratization - 1)  # d_a
        xx = np.arange(-a, a + q, q)

        output_forces = np.zeros(2)
        p_x = np.zeros((fastsim_patch.discratization, fastsim_patch.discratization))
        p_y = np.zeros((fastsim_patch.discratization, fastsim_patch.discratization))
        s_out = np.zeros((fastsim_patch.discratization, fastsim_patch.discratization))
        g_bound = np.zeros((fastsim_patch.discratization, fastsim_patch.discratization))
        s_x = np.zeros((fastsim_patch.discratization, fastsim_patch.discratization))
        s_y = np.zeros((fastsim_patch.discratization, fastsim_patch.discratization))

        for jj in range(fastsim_patch.discratization):
            a_y = a * np.sqrt(1 - (yy[jj] / b) ** 2)
            c_x = (slip_state.nu_x - slip_state.phi * yy[jj]) * state_dyn.state_v_x

            p_1x = 0
            p_1y = 0

            for kk in range(fastsim_patch.discratization - 1, -1, -1):
                if abs(xx[kk]) <= a_y and a_y > 1e-11:
                    g_bound[kk, jj] = (
                        (2 * contact_patch.normal_force / np.pi / a**3 / b)
                        * material_properties.friction
                        * (a_y**2 - xx[kk] ** 2)
                    )
                    c_y = (
                        slip_state.nu_y + slip_state.phi * xx[kk]
                    ) * state_dyn.state_v_x

                    p_x_lin = p_1x - q * (
                        slip_state.nu_x / L_x - slip_state.phi * yy[jj] / L_phi
                    )
                    p_y_lin = p_1y - q * (
                        slip_state.nu_y / L_y + slip_state.phi * xx[kk] / L_phi
                    )
                    p_x[kk, jj] = p_x_lin
                    p_y[kk, jj] = p_y_lin
                    P_sat = np.sqrt(p_x_lin**2 + p_y_lin**2)

                    if P_sat >= g_bound[kk, jj]:
                        
                        p_x[kk, jj] = g_bound[kk, jj] / P_sat * p_x_lin
                        p_y[kk, jj] = g_bound[kk, jj] / P_sat * p_y_lin

                        s_x[kk, jj] = c_x + L_x * state_dyn.state_v_x / q * (
                            p_x[kk, jj] - p_1x
                        )
                        s_y[kk, jj] = c_y + L_y * state_dyn.state_v_x / q * (
                            p_y[kk, jj] - p_1y
                        )

                        P_sat = np.sqrt(p_x[kk, jj] ** 2 + p_y[kk, jj] ** 2)
                        s_out[kk, jj] = (
                            np.sqrt(s_x[kk, jj] ** 2 + s_y[kk, jj] ** 2)
                            / state_dyn.state_v_x
                        ) * q

                    p_1x = p_x[kk, jj]
                    p_1y = p_y[kk, jj]

                    output_forces += q * r * np.array([p_1x, p_1y])

        fastsim_patch.xx = xx
        fastsim_patch.yy = yy
        fastsim_patch.F_x = output_forces[0]
        fastsim_patch.F_y = output_forces[1]
        fastsim_patch.p_x = p_x
        fastsim_patch.p_y = p_y
        fastsim_patch.s_x = s_x
        fastsim_patch.s_y = s_y
        fastsim_patch.s_out = s_out
        fastsim_patch.g_bound = g_bound

    else:
        raise TypeError("Error in FASTSIM calculation: not an equivalent ellipse normal contact method")
