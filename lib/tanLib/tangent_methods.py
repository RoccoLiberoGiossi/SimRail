from lib.contact_settings import *
from lib.normLib.contact_methods import eqv_el_normal_contact, kik_pio_normal_contact
from lib.contact_material import material, kalker_coefficients
from lib.dynLib.contact_dyn_state import dynamic_state
from lib.jittable_functions import compute_fastsim_core

class tangent_creepage:
    def __init__(self, nu_x=0.0, nu_y=0.0, phi=0.0):
        self.nu_x = nu_x
        self.nu_y = nu_y
        self.phi = phi

#TODO: check spelling of discratization
# class fastsim:
#     def __init__(self, discratization=58):
#         self.discratization = discratization
#         self.F_x = 0.0
#         self.F_y = 0.0
#         self.xx = np.zeros(1)
#         self.yy = np.zeros(1)
#         self.p_x = np.zeros((self.discratization, self.discratization))
#         self.p_y = np.zeros((self.discratization, self.discratization))
#         self.s_x = np.zeros((self.discratization, self.discratization))
#         self.s_y = np.zeros((self.discratization, self.discratization))
#         self.s_out = np.zeros((self.discratization, self.discratization))
#         self.g_bound = np.zeros((self.discratization, self.discratization))

#     def reset_to_zero(self):
#         for attribute in vars(self):
#             setattr(self, attribute, 0)

#     def calc_tangent_force(self, contact_patch, material_properties, kalker_tables, slip_state, state_dyn):
        
#         if isinstance(contact_patch, eqv_el_normal_contact):

#             if not isinstance(self, fastsim):
#                 raise TypeError("Error in FASTSIM calculation: not a fastsim method provided")
#             if not isinstance(material_properties, material):
#                 raise TypeError("Error in FASTSIM calculation: not a material method")
#             if not isinstance(state_dyn, dynamic_state):
#                 raise TypeError(
#                     "Error in FASTSIM calculation: not a correct dynamic state method"
#                 )
#             if not isinstance(slip_state, tangent_creepage):
#                 raise TypeError(
#                     "Error in FASTSIM calculation: not a correct dynamic state method"
#                 )
#             if not isinstance(kalker_tables, kalker_coefficients):
#                 raise TypeError(
#                     "Error in FASTSIM calculation: not a correct dynamic state method"
#                 )

#             a = contact_patch.semi_axis_a
#             b = contact_patch.semi_axis_b

#             # Interpolating values
#             c_11 = kalker_tables.pchip_interp_c_11(a / b)
#             c_22 = kalker_tables.pchip_interp_c_22(a / b)
#             c_23 = kalker_tables.pchip_interp_c_23(a / b)

#             # Calculating L_x, L_y, and L_phi
#             L_x = 8 * a / (3 * material_properties.G * c_11)
#             L_y = 8 * a / (3 * material_properties.G * c_22)
#             L_phi = np.pi * a**2 / (4 * material_properties.G * np.sqrt(a * b) * c_23)

#             # Calculating r and q
#             r = 2 * b / (self.discratization - 1)  # d_b
#             yy = np.arange(-b, b + r, r)
#             q = 2 * a / (self.discratization - 1)  # d_a
#             xx = np.arange(-a, a + q, q)

#             output_forces = np.zeros(2)

#             for jj in range(self.discratization):
                
#                 a_y = a * np.sqrt(1 - (yy[jj] / b) ** 2) if abs(yy[jj]) <= b else 0
#                 c_x = (slip_state.nu_x - slip_state.phi * yy[jj]) * state_dyn.state_v_x

#                 p_1x = 0
#                 p_1y = 0

#                 for kk in range(self.discratization - 1, -1, -1):
#                     if abs(xx[kk]) <= a_y and a_y > 1e-11:
#                         self.g_bound[kk, jj] = (
#                             (2 * contact_patch.normal_force / np.pi / a**3 / b)
#                             * material_properties.friction
#                             * (a_y**2 - xx[kk] ** 2)
#                         )
#                         c_y = (
#                             slip_state.nu_y + slip_state.phi * xx[kk]
#                         ) * state_dyn.state_v_x

#                         p_x_lin = p_1x - q * (
#                             slip_state.nu_x / L_x - slip_state.phi * yy[jj] / L_phi
#                         )
#                         p_y_lin = p_1y - q * (
#                             slip_state.nu_y / L_y + slip_state.phi * xx[kk] / L_phi
#                         )
#                         self.p_x[kk, jj] = p_x_lin
#                         self.p_y[kk, jj] = p_y_lin
#                         P_sat = np.sqrt(p_x_lin**2 + p_y_lin**2)

#                         if P_sat >= self.g_bound[kk, jj]:
                            
#                             self.p_x[kk, jj] = self.g_bound[kk, jj] / P_sat * p_x_lin
#                             self.p_y[kk, jj] = self.g_bound[kk, jj] / P_sat * p_y_lin

#                             self.s_x[kk, jj] = c_x + L_x * state_dyn.state_v_x / q * (
#                                 self.p_x[kk, jj] - p_1x
#                             )
#                             self.s_y[kk, jj] = c_y + L_y * state_dyn.state_v_x / q * (
#                                 self.p_y[kk, jj] - p_1y
#                             )

#                             P_sat = np.sqrt(self.p_x[kk, jj] ** 2 + self.p_y[kk, jj] ** 2)
#                             self.s_out[kk, jj] = (
#                                 np.sqrt(self.s_x[kk, jj] ** 2 + self.s_y[kk, jj] ** 2)
#                                 / state_dyn.state_v_x
#                             ) * q

#                         p_1x = self.p_x[kk, jj]
#                         p_1y = self.p_y[kk, jj]

#                         output_forces += q * r * np.array([p_1x, p_1y])

#             self.xx = xx
#             self.yy = yy
#             self.F_x = output_forces[0]
#             self.F_y = output_forces[1]

#         elif isinstance(contact_patch, kik_pio_normal_contact):
#             raise NotImplementedError(
#                 "Kik-Pio normal contact method is not implemented for tangent force calculation yet."
#             )

#         else:
#             raise TypeError("Error in FASTSIM calculation: not an equivalent ellipse normal contact method")
        
#TODO: check spelling of discratization
class fastsim:
    def __init__(self, discratization=58):
        self.discratization = discratization
        self.F_x = 0.0
        self.F_y = 0.0
        self.xx = np.zeros(1)
        self.yy = np.zeros(1)
        self.p_x = np.zeros((self.discratization, self.discratization))
        self.p_y = np.zeros((self.discratization, self.discratization))
        self.s_x = np.zeros((self.discratization, self.discratization))
        self.s_y = np.zeros((self.discratization, self.discratization))
        self.s_out = np.zeros((self.discratization, self.discratization))
        self.g_bound = np.zeros((self.discratization, self.discratization))
    def reset_to_zero(self):
        for attribute in vars(self):
            setattr(self, attribute, 0)
    def calc_tangent_force(self, contact_patch, material_properties, kalker_tables, slip_state, state_dyn):
    
        if isinstance(contact_patch, eqv_el_normal_contact):
            if not isinstance(self, fastsim):
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
            if not isinstance(kalker_tables, kalker_coefficients):
                raise TypeError(
                    "Error in FASTSIM calculation: not a correct dynamic state method"
                )
            a = contact_patch.semi_axis_a
            b = contact_patch.semi_axis_b
            # Interpolating values
            c_11 = kalker_tables.pchip_interp_c_11(a / b)
            c_22 = kalker_tables.pchip_interp_c_22(a / b)
            c_23 = kalker_tables.pchip_interp_c_23(a / b)
            # Calculating L_x, L_y, and L_phi
            L_x = 8 * a / (3 * material_properties.G * c_11)
            L_y = 8 * a / (3 * material_properties.G * c_22)
            L_phi = np.pi * a**2 / (4 * material_properties.G * np.sqrt(a * b) * c_23)
            # Calculating r and q
            r = 2 * b / (self.discratization - 1)  # d_b
            yy = np.arange(-b, b + r, r)
            q = 2 * a / (self.discratization - 1)  # d_a
            xx = np.arange(-a, a + q, q)
            # Call JIT-compiled core
            results = compute_fastsim_core(
                xx, yy, q, r, a, b,
                contact_patch.normal_force,
                material_properties.friction,
                L_x, L_y, L_phi,
                slip_state.nu_x,
                slip_state.nu_y,
                slip_state.phi,
                state_dyn.state_v_x,
                self.discratization
            )
        
            # Unpack results
            (self.p_x, self.p_y, self.s_x, self.s_y, 
            self.s_out, self.g_bound, self.F_x, self.F_y) = results
        
            self.xx = xx
            self.yy = yy
        elif isinstance(contact_patch, kik_pio_normal_contact):
            raise NotImplementedError(
                "Kik-Pio normal contact method is not implemented for tangent force calculation yet."
            )
        else:
            raise TypeError("Error in FASTSIM calculation: not an equivalent ellipse normal contact method")