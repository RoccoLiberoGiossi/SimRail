import numpy as np
from lib.contact_settings import *
from lib.contact_dyn_state import dynamic_state
from lib.contact_solver import patch_search

from lib.contact_wheel_rail import wheel, rail
from lib.contact_material import material
from lib.contact_methods import eqv_el_normal_contact


# TODO move this from here. It should be in a more globally accessible place
def calc_global_force(contact_patches):
    return sum(patch.Q_force for patch in contact_patches)


def iterative_Q_search(
    Dz0,
    wheel_state,
    Wheel,
    Rail,
    material_model,
    contact_patches,
    discretization=58,
    Q_solve=50000,
):
    wheel_state.state_z = Dz0
    Wheel.set_dynamic_state(wheel_state)
    Wheel.calculate_position()
    patch_search(Wheel, Rail, material_model, contact_patches, discretization)

    total_force = calc_global_force(contact_patches)

    return np.abs(total_force - Q_solve)


def static_contact(
    deltays,
    Wheel,
    Rail,
    contact_patches,
    material_model,
    discretization=58,
    Q_solve=50000,
    tollerance=50,
):
    if not isinstance(Wheel, wheel):
        raise TypeError("Error in contact forces calculation: wheel not given")

    if not isinstance(Rail, rail):
        raise TypeError("Error in contact forces calculation: rail not given")

    if not isinstance(material_model, material):
        raise TypeError("Error in contact forces calculation: no material provided")

    result_state = {}
    for jj in range(len(contact_patches)):
        result_state[f"patch_{jj}"] = []

    new_state = dynamic_state()

    dz0_prew = 0

    for deltyIndx, deltay in enumerate(deltays):

        new_state.state_y = deltay
        Wheel.set_dynamic_state(new_state)
        Wheel.calculate_position()

        if deltyIndx == 0 and Wheel.lr == RIGHT:
            Dz0 = 0.002
        elif deltyIndx == 0 and Wheel.lr == LEFT:
            Dz0 = 0.008
        else:
            Dz0 = dz0_prew

        root = fsolve(
            iterative_Q_search,
            Dz0,
            args=(new_state, Wheel, Rail, material_model, contact_patches),
        )
        Qout = calc_global_force(contact_patches)
        Dz0 = root

        iteration = 1

        while (np.abs(Qout - Q_solve) > tollerance) and (iteration < 100):
            if (Qout - Q_solve) > 0:
                root = fsolve(
                    iterative_Q_search,
                    Dz0 * (1 + (100 - iteration) / 10000),
                    args=(new_state, Wheel, Rail, material_model, contact_patches),
                )
            else:
                root = fsolve(
                    iterative_Q_search,
                    Dz0 * (1 - (100 - iteration) / 10000),
                    args=(new_state, Wheel, Rail, material_model, contact_patches),
                )
            Qout = calc_global_force(contact_patches)
            # Dz0 = root
            iteration = iteration + 1

        iterative_Q_search(
            root,
            new_state,
            Wheel,
            Rail,
            material_model,
            contact_patches,
            discretization,
            Q_solve,
        )

        dz0_prew = root

        # result_state_0 = [
        #     float(root),
        #     float(contact_patches[0].normal_force),
        #     float(contact_patches[0].Q_force),
        #     float(contact_patches[0].approach),
        #     float(contact_patches[0].semi_axis_a),
        #     float(contact_patches[0].semi_axis_b),
        # ]

        for jj in range(len(contact_patches)):
            result_state_patch = [
                float(root),
                float(contact_patches[jj].normal_force),
                float(contact_patches[jj].Q_force),
                float(contact_patches[jj].Y_force),
                float(contact_patches[jj].approach),
                float(contact_patches[jj].semi_axis_a),
                float(contact_patches[jj].semi_axis_b),
            ]
            result_state[f"patch_{jj}"].append(result_state_patch)

        # result_state.append(result_state_0)

    for jj in range(len(contact_patches)):
        result_state[f"patch_{jj}"] = np.array(result_state[f"patch_{jj}"])

    return result_state
