from lib.contact_settings import *

# --------------------------------------------------------- #
# Setting class to share material properties                #
# --------------------------------------------------------- #

class material:
    def __init__(self, Elastic_Modulus=210e9, Shear_Modulus=0.28, friction=0.4):
        self.Elastic_Modulus = Elastic_Modulus
        self.Shear_Modulus = Shear_Modulus
        # this must be modified in case the wheel and the rail have two different materials
        # this supposed that wheel and rail are the same
        self.Estar = 1 / ((1 + self.Shear_Modulus**2) / self.Elastic_Modulus * 2)
        self.kik_pyo_constant = np.pi*self.Estar/2/(1-self.Shear_Modulus**2)
        self.G = self.Elastic_Modulus / (2 * (1 + self.Shear_Modulus))
        self.friction = friction