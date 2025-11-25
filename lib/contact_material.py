from lib.contact_settings import *
from lib.contact_kalkers_table import *

# --------------------------------------------------------- #
# Setting class to share material properties                #
# --------------------------------------------------------- #

class material:
    def __init__(self, Elastic_Modulus=210e9, Shear_Modulus=0.28, friction=0.4):
        self.Elastic_Modulus = Elastic_Modulus
        self.Shear_Modulus = Shear_Modulus
        self.friction = friction
        # this must be modified in case the wheel and the rail have two different materials
        # this supposed that wheel and rail are the same
        self.Estar = 1 / ((1 + self.Shear_Modulus**2) / self.Elastic_Modulus * 2)
        self.kik_pio_constant = np.pi*self.Estar/2/(1-self.Shear_Modulus**2)
        self.G = self.Elastic_Modulus / (2 * (1 + self.Shear_Modulus))

class kalker_coefficients:
    def __init__(self, material_properties : material):
        self.c11_share_interpolator = PchipInterpolator(share_modulus, c11tab)
        self.c22_share_interpolator = PchipInterpolator(share_modulus, c22tab)
        self.c23_share_interpolator = PchipInterpolator(share_modulus, c23tab)
        self.c33_share_interpolator = PchipInterpolator(share_modulus, c33tab)

        self.c11_tab = self.c11_share_interpolator(material_properties.Shear_Modulus)
        self.c22_tab = self.c22_share_interpolator(material_properties.Shear_Modulus)
        self.c23_tab = self.c23_share_interpolator(material_properties.Shear_Modulus)
        self.c33_tab = self.c33_share_interpolator(material_properties.Shear_Modulus)

        self.pchip_interp_c_11 = PchipInterpolator(abratio, self.c11_tab)
        self.pchip_interp_c_22 = PchipInterpolator(abratio, self.c22_tab)
        self.pchip_interp_c_23 = PchipInterpolator(abratio, self.c23_tab)
        self.pchip_interp_c_33 = PchipInterpolator(abratio, self.c33_tab)