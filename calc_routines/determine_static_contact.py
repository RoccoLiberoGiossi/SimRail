import numpy as np
from lib.contact_settings import *
from lib.dynLib.contact_dyn_state import dynamic_state
from lib.normLib.contact_solver import NormalContactSolver
from tqdm import tqdm

from lib.geoLib.contact_wheel_rail import wheel, rail
from lib.contact_material import material
from lib.normLib.contact_methods import eqv_el_normal_contact, kik_pio_normal_contact


# TODO move this from here. It should be in a more globally accessible place
def calc_global_force(contact_patches):
    return sum(patch.Q_force for patch in contact_patches)


def iterative_Q_search(
    Dz0,
    wheel_state,
    Wheel: wheel,
    Rail: rail,
    contact_model: NormalContactSolver,
    material_model: material,
    discretization=58,
    Q_solve=50000,
):
    wheel_state.state_z = Dz0
    Wheel.set_dynamic_state(wheel_state)
    Wheel.calculate_position()
    contact_model.patch_search(Wheel, Rail, material_model, discretization)

    total_force = calc_global_force(contact_model.contact_patches)

    return np.abs(total_force - Q_solve)


def static_contact(
    deltays: np.ndarray[float],
    Wheel: wheel,
    Rail: rail,
    contact_model: NormalContactSolver,
    material_model: material,
    discretization: int = 58,
    Q_solve: float = 50000,
    tollerance: float = 50,
    max_iter: int = 100,
    method: str = "hybr",
):
    if not isinstance(Wheel, wheel):
        raise TypeError("Error in contact forces calculation: wheel not given")

    if not isinstance(Rail, rail):
        raise TypeError("Error in contact forces calculation: rail not given")

    if not isinstance(material_model, material):
        raise TypeError("Error in contact forces calculation: no material provided")

    result_state = {}
    result_state_pressure = {}

    if isinstance(contact_model.contact_patches[0], eqv_el_normal_contact):
        for jj in range(len(contact_model.contact_patches)):
            result_state[f"patch_{jj}"] = np.zeros((len(deltays), 8), dtype=np.float64)
    elif isinstance(contact_model.contact_patches[0], kik_pio_normal_contact):
        for jj in range(len(contact_model.contact_patches)):
            result_state[f"patch_{jj}"] = np.zeros((len(deltays), 7), dtype=np.float64)
            result_state_pressure[f"patch_{jj}"] = np.zeros(
                (len(deltays) * discretization, discretization + 2), dtype=np.float64
            )

    new_state = dynamic_state()

    dz0_prew = 0

    for deltyIndx, deltay in tqdm(enumerate(deltays), total=len(deltays)):

        new_state.state_y = deltay
        Wheel.set_dynamic_state(new_state)
        Wheel.calculate_position()

        if deltyIndx == 0 and Wheel.lr == RIGHT:
            Dz0 = 0.002
        elif deltyIndx == 0 and Wheel.lr == LEFT:
            Dz0 = 0.008
        else:
            Dz0 = dz0_prew

        solution = root(
            iterative_Q_search,
            Dz0,
            args=(new_state, Wheel, Rail, contact_model, material_model, discretization, Q_solve),
            method=method,
        )
        # print(solution)
        # solution = fsolve(
        #     iterative_Q_search,
        #     Dz0,
        #     args=(new_state, Wheel, Rail, contact_model, material_model, discretization, Q_solve),
        # )
        Qout = calc_global_force(contact_model.contact_patches)
        Dz0 = solution.x

        iteration = 1

        while (np.abs(Qout - Q_solve) > tollerance) and (iteration < max_iter):
            if (Qout - Q_solve) > 0:
                solution = root(
                    iterative_Q_search,
                    Dz0 * (1 + (100 - iteration) / 10000),
                    args=(new_state, Wheel, Rail, contact_model, material_model, discretization, Q_solve),
                    method=method,
                )
            else:
                solution = root(
                    iterative_Q_search,
                    Dz0 * (1 - (100 - iteration) / 10000),
                    args=(new_state, Wheel, Rail, contact_model, material_model, discretization, Q_solve),
                    method=method,
                )
            Qout = calc_global_force(contact_model.contact_patches)
            iteration = iteration + 1

        iterative_Q_search(
            solution.x,
            new_state,
            Wheel,
            Rail,
            contact_model,
            material_model,
            discretization,
            Q_solve,
        )

        dz0_prew = solution.x

        # if isinstance(contact_model, eqv_el_normal_contact):

        for jj in range(len(contact_model.contact_patches)):
            if isinstance(contact_model.contact_patches[jj], eqv_el_normal_contact):
                result_state[f"patch_{jj}"][deltyIndx] = [
                    float(solution.x),
                    float(contact_model.contact_patches[jj].normal_force),
                    float(contact_model.contact_patches[jj].Q_force),
                    float(contact_model.contact_patches[jj].Y_force),
                    float(contact_model.contact_patches[jj].approach),
                    float(contact_model.contact_patches[jj].centroid_wheel),
                    float(contact_model.contact_patches[jj].semi_axis_a),
                    float(contact_model.contact_patches[jj].semi_axis_b),
                ]
            elif isinstance(contact_model.contact_patches[jj], kik_pio_normal_contact):
                result_state[f"patch_{jj}"][deltyIndx] = [
                    float(solution.x),
                    float(contact_model.contact_patches[jj].normal_force),
                    float(contact_model.contact_patches[jj].Q_force),
                    float(contact_model.contact_patches[jj].Y_force),
                    float(contact_model.contact_patches[jj].approach),
                    float(contact_model.contact_patches[jj].centroid_wheel),
                    float(contact_model.contact_patches[jj].pressure_0),
                ]
                # TODO: for god sake fix this part!
                if contact_model.contact_patches[jj].normal_force > 0:
                    data_to_stack = np.hstack(
                        [
                            contact_model.contact_patches[jj].x_patch[:, np.newaxis],
                            contact_model.contact_patches[jj].y_patch[:, np.newaxis],
                            contact_model.contact_patches[jj].pressure_patch,
                        ]
                    )
                    result_state_pressure[f"patch_{jj}"][
                        discretization * (deltyIndx) : discretization * (deltyIndx + 1),
                        0 : discretization + 2,
                    ] = data_to_stack

    for jj in range(len(contact_model.contact_patches)):
        result_state[f"patch_{jj}"] = np.array(result_state[f"patch_{jj}"])

    return result_state, result_state_pressure
