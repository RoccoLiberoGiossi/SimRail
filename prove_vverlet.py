from lib.contact_settings import *
from lib.contact_material import material
from lib.geoLib.contact_wheel_rail import wheel, rail
from lib.dynLib.contact_dyn_state import dynamic_state
from lib.normLib.contact_methods import eqv_el_normal_contact, kik_pio_normal_contact
from lib.normLib.contact_solver import NormalContactSolver
from lib.tanLib.tangent_methods import fastsim, tangent_creepage
from lib.contact_material import kalker_coefficients
from calc_routines.determine_static_contact import static_contact

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time


def Rx(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def Ry(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def Rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def contact_local_to_global(F_x, F_y, Q_force, Y_force, yaw, roll, contact_angle):
    """
    Map force vector from local contact plane to global frame.

    Inputs:
    F_x : float          -> Longitudinal force
    F_y : float          -> Lateral force
    Q_force : float      -> Vertical force
    Y_force : float      -> Additional lateral force
    yaw     : psi   (rad) -> q[4]
    roll    : phi   (rad) -> q[5]
    contact_angle : alpha (rad) -> contacts.contact_angle

    Returns:
    F_global : numpy array (3,)
    """

    F_y_local = F_y * np.sin(contact_angle)
    F_z_local = F_y * np.cos(contact_angle)

    # F_local = [F_x, Y_force, F_z_local + Q_force]
    F_local = [F_x, F_y_local + Y_force, F_z_local + Q_force]

    F_local = np.asarray(F_local, dtype=float).reshape(3)

    R_wheel = Rz(yaw) @ Rx(roll)

    # print(R_wheel @ F_local)

    # return F_local
    return R_wheel @ F_local


def dynamics(
    t,
    x,
    wheels: wheel,
    rails: rail,
    contact_eq_s: list[NormalContactSolver],
    material_model,
    tangent_method,
    kalker_tables,
    wheelset_mass,
    rail_y,
    Y_forces_series=[],
    Q_forces_series=[],
    rail_position = [],
    Wheelset_y_position = [],
    Wheelset_z_position = [],
):

    dx = np.zeros_like(x)
    q = x[0:6]
    v = x[6:12]

    dx[0:6] = v

    new_state = dynamic_state(
        state_x=q[0],
        state_y=q[1],
        state_z=q[2],
        state_pitch=q[3],
        state_yaw=q[4],
        state_roll=q[5],
        state_v_x=v[0],
        state_v_y=v[1],
        state_v_z=v[2],
        state_v_pitch=v[3],
        state_v_yaw=v[4],
        state_v_roll=v[5],
    )

    F_vector = np.zeros(6)

    Y_forces = []
    Q_forces = []

    for index, wheel in enumerate(wheels):

        wheel.set_dynamic_state(new_state)
        wheel.calculate_position()

        y_rail = rail_y(t)
        rails[index].calculate_position(y_rail, 0)

        contact_eq_s[index].patch_search(wheel, rails[index], material_model, discretization=30)
        # radius_inerpolator = PchipInterpolator(
        #     wheel.wheel_radius[:, 0], wheel.wheel_radius[:, 1]
        # )

        Fx_forces = []
        Fy_forces = []
        moments_pitch = []
        moments_yaw = []
        moments_roll = []
        creep_xs = []
        creep_ys = []
        phi_s = []
        delta_rs = []
        centroids = []

        for index, contacts in enumerate(contact_eq_s[index].contact_patches):
            contact_radius = contacts.Rlocal
            delta_r = wheel.r0 - contact_radius
            if type(contacts.centroid_wheel) != int:
                print("in here")
                if wheel.lr == RIGHT:
                    nu_x = -(
                        delta_r / wheel.r0
                        + np.abs(contacts.centroid_wheel.item()) / 2 * v[4] / v[0]
                    )
                elif wheel.lr == LEFT:
                    nu_x = -(
                        delta_r / wheel.r0
                        - np.abs(contacts.centroid_wheel.item()) / 2 * v[4] / v[0]
                    )

                nu_y = 1 / np.cos(contacts.contact_angle) * (v[1] / v[0] - q[4])
                phi = -np.cos(contacts.contact_angle) / wheel.r0 + v[4] / v[0] * np.cos(
                    contacts.contact_angle
                )

                creep = tangent_creepage(nu_x=nu_x, nu_y=nu_y, phi=phi)

                tangent_method.calc_tangent_force(
                    contacts, material_model, kalker_tables, creep, new_state
                )

                # force_vector = contact_local_to_global(
                #     0,
                #     tangent_method.F_y,
                #     contacts.Q_force,
                #     0,
                #     0,
                #     0,
                #     0,
                # )
                force_vector = contact_local_to_global(
                    tangent_method.F_x,
                    tangent_method.F_y,
                    contacts.Q_force,
                    contacts.Y_force,
                    q[4],
                    q[5],
                    contacts.contact_angle,
                )
                # force_vector = contact_local_to_global(
                #     0.0,
                #     0.0,
                #     contacts.Q_force,
                #     contacts.Y_force,
                #     q[4],
                #     q[5],
                #     contacts.contact_angle,
                # )

                Y_forces.append(contacts.Y_force)
                Q_forces.append(contacts.Q_force)
                Fx_forces.append(tangent_method.F_x)
                Fy_forces.append(tangent_method.F_y)
                moments_pitch.append(contact_radius * tangent_method.F_x)
                moments_yaw.append(contacts.centroid_wheel.item() * tangent_method.F_x)
                moments_roll.append(contacts.centroid_wheel.item() * contacts.Q_force)
                creep_xs.append(nu_x)
                creep_ys.append(nu_y)
                phi_s.append(phi)
                delta_rs.append(delta_r)
                centroids.append(contacts.centroid_wheel.item())
                print(contacts.Y_force)
            else:
                force_vector = contact_local_to_global(
                    0.0,
                    0.0,
                    contacts.Q_force,
                    contacts.Y_force,
                    q[4],
                    q[5],
                    contacts.contact_angle,
                )
                Y_forces.append(contacts.Y_force)
                Q_forces.append(contacts.Q_force)

            

            # F_vector[0] += 0#contacts.Q_force
            # F_vector[1] += contacts.Y_force
            # F_vector[2] += contacts.Q_force
            # F_vector[3] += 0#contact_radius.item() * force_vector[0]
            # F_vector[4] += 0#contacts.centroid_wheel.item() * force_vector[0]
            # F_vector[5] += 0#contacts.centroid_wheel.item() * force_vector[2]
            F_vector[0] += force_vector[0]
            F_vector[1] += force_vector[1]
            F_vector[2] += force_vector[2]
            F_vector[3] += contact_radius * force_vector[0]
            F_vector[4] += contacts.centroid_wheel.item() * force_vector[0] if type(contacts.centroid_wheel) != int else 0
            F_vector[5] += contacts.centroid_wheel.item() * force_vector[2] if type(contacts.centroid_wheel) != int else 0

        # if wheel.lr == RIGHT:
        #     print(f"Wheel Right Forces:")
        # elif wheel.lr == LEFT:
        #     print(f"Wheel Left Forces:")
        # print(f"centroid: {centroids}")
        # print(f"Y forces: {Y_forces}")
        # print(f"Q forces: {Q_forces}")
        # print(f"Fx forces: {Fx_forces}")
        # print(f"Fy forces: {Fy_forces}")
        # print(f"Moments pitch: {moments_pitch}")
        # print(f"Moments yaw: {moments_yaw}")
        # print(f"Moments roll: {moments_roll}")
        # print(f"g force:{wheelset_mass[0] * 9.81}")
        # print(f"Creep x: {creep_xs}")
        # print(f"Creep y: {creep_ys}")
        # print(f"Phi: {phi_s}")
        # print(f"Delta r: {delta_rs}")

    Y_forces_series.append(Y_forces)
    Q_forces_series.append(Q_forces)
    rail_position.append(y_rail)
    Wheelset_y_position.append(new_state.state_y)
    Wheelset_z_position.append(new_state.state_z)

    F_vector[2] -= wheelset_mass[0] * 9.81
    print(f"Total Force Vector: {F_vector}")
    dx[6:12] = -F_vector / wheelset_mass

    print(f"doing...{t}")

    global _fig, _ax_1, _ax_2, _ax_3, _ax_4, _ax_5, _ax_6

    # Create the figure only once
    if _fig is None:
        # _fig, _ax_1 = plt.subplots(1, 1, figsize=(10, 4))
        _fig, axes = plt.subplots(2, 3, figsize=(10, 4))
        _ax_1, _ax_2, _ax_3, _ax_4, _ax_5, _ax_6 = axes.flatten()
        _ax_1.set_title(f"Wheel {t} s")
        _ax_1.set_aspect("equal")
        _ax_2.set_title("Y Forces")
        _ax_3.set_title("Q Forces")
        _ax_4.set_title("Rail Position")
        _ax_5.set_title("Wheelset y Position")
        _ax_6.set_title("Rail Position")
        # _ax_2.set_aspect("equal")
        plt.show()

    # Clear the old plots
    _ax_1.cla()
    _ax_2.cla()
    _ax_3.cla()
    _ax_4.cla()
    _ax_5.cla()
    _ax_6.cla()

    # Plot wheel geometry
    _ax_1.plot(rails[1].rail_profile_pos[:, 0], rails[1].rail_profile_pos[:, 1], c="r")
    _ax_1.plot(wheels[1].wheel_profile_pos[:, 0], wheels[1].wheel_profile_pos[:, 1], c="black")
    _ax_1.plot(rails[0].rail_profile_pos[:, 0], rails[0].rail_profile_pos[:, 1], c="r")
    _ax_1.plot(wheels[0].wheel_profile_pos[:, 0], wheels[0].wheel_profile_pos[:, 1], c="black")
    _ax_1.set_title(f"Wheel {t} s")
    _ax_1.set_aspect("equal")

    # Plot rail geometry (select correct rail)
    _ax_2.plot(Y_forces_series)
    _ax_2.grid(True)
    _ax_2.set_title("Y Forces")
    # _ax_2.set_aspect("equal")

    # Plot Q forces
    _ax_3.plot(Q_forces_series)
    _ax_3.grid(True)
    _ax_3.set_title("Q Forces")
    
    # Plot rail pos
    _ax_4.plot(rail_position)
    _ax_4.grid(True)
    _ax_4.set_title("Rail Position")
    
    # Plot rail pos
    _ax_5.plot(Wheelset_y_position)
    _ax_5.grid(True)
    _ax_5.set_title("Wheelset y Position")

    # Plot rail pos
    _ax_6.plot(Wheelset_z_position)
    _ax_6.grid(True)
    _ax_6.set_title("Wheelset z Position")
    # _ax_3.set_aspect("equal")

    # Refresh only
    _fig.canvas.draw()
    _fig.canvas.flush_events()

    return dx


def velocity_verlet_step(
    x,
    a_old,
    dt,
    t,
    force_func,
    wheels,
    rails,
    contact_eqs,
    material_model,
    fastsim_patch,
    kalker_tables,
    mass_vector,
    rail_inerpolator,
):
    x_old = x[0:6]
    v_old = x[6:12]
    x_new_star = x_old + v_old * dt + 0.5 * a_old * dt**2
    # x_new = np.stack((x_new_star, x[6:12]), axis=0).flatten()
    x_new = np.concatenate([x_new_star, v_old])
    a_star = force_func(
        t,
        x_new,
        wheels,
        rails,
        contact_eqs,
        material_model,
        fastsim_patch,
        kalker_tables,
        mass_vector,
        rail_inerpolator,
    )
    a_new = a_star[6:12]
    v_new = v_old + 0.5 * (a_old + a_new) * dt
    return x_new_star, v_new, a_new

wheels = [wheel(), wheel(lr=LEFT)]
rails = [rail(), rail(lr=LEFT)]

n_patches = 3
contact_eqs = [
    NormalContactSolver(eqv_el_normal_contact(), n_patches),
    NormalContactSolver(eqv_el_normal_contact(), n_patches),
]

material_model = material()

kalker_tables = kalker_coefficients(material_properties=material_model)
fastsim_patch = fastsim(discratization=30)

wheelset_mass = 780.0
I_x = 80.0
I_y = 350.0
I_z = 350.0

patches_per_deltays_eq, result_state_pressure_eq = static_contact(
    [0],
    wheels[0],
    rails[0],
    contact_eqs[0],
    material_model,
    Q_solve=wheelset_mass * 9.81 / 2,
    discretization=30
)

print(patches_per_deltays_eq["patch_0"][0][0])

# patches_per_deltays_eq, result_state_pressure_eq = static_contact(
#     [0],
#     wheels[1],
#     rails[1],
#     contact_eqs[1],
#     material_model,
#     Q_solve=wheelset_mass * 9.81 / 2,
# )

print(patches_per_deltays_eq["patch_0"][0][0])
input()

x0 = np.zeros(12)
x0[2] = patches_per_deltays_eq["patch_0"][0][0]
x0[6] = 10 / 3.6

mass_vector = np.array([wheelset_mass, wheelset_mass, wheelset_mass, I_x, I_y, I_z])

tspan = (0, 2)
dt = 0.1e-3

t_eval = np.arange(tspan[0], tspan[1] + dt, dt)

ds = 0.05e-3
rail_y = np.zeros_like(t_eval)
rail_y[len(t_eval)//4:len(t_eval)//2] = ds*np.arange(len(t_eval)//4)/(len(t_eval)//4)
rail_y[len(t_eval)//2:len(t_eval)//4*3] = ds*np.flip(np.arange(len(t_eval)//4))/(len(t_eval)//4)

rail_y_interpolator = PchipInterpolator(t_eval, rail_y)
fig, ax = plt.subplots()
ax.plot(t_eval, rail_y)
plt.show()

input()

plt.ion()

_fig = None
_ax_1 = None
_ax_2 = None
_ax_3 = None
_ax_4 = None
_ax_5 = None
_ax_6 = None

# sol = solve_ivp(
#     dynamics,
#     tspan,
#     x0,
#     t_eval=t_eval,
#     method="RK45",
#     max_step=dt,
#     args=(
#         wheels,
#         rails,
#         contact_eqs,
#         material_model,
#         fastsim_patch,
#         kalker_tables,
#         mass_vector,
#         rail_y_interpolator,
#     ),  # pass extra parameters
# )

# plt.ioff()  # Turn off interactive mode
# plt.show()  # Keep final plot open

# fig, axes = plt.subplots(3, 2, figsize=(12, 8))
# axes = axes.flatten()
# for i, ax in enumerate(axes):
#     ax.plot(sol.t, sol.y[i, :])

# fig.tight_layout()
# plt.show()

# # Enable interactive mode
# plt.ion()

fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()  # Flatten to 1D array for easy indexing

names = ["x", "y", "z", "pitch", "yaw", "roll"]
# Initialize empty lines for each subplot
lines = []
for i, ax in enumerate(axes):
    (line,) = ax.plot([], [], "b-", linewidth=2)
    lines.append(line)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"x[{i}]")
    ax.set_title(f"State Variable {names[i]}")
    ax.grid(True)

fig.tight_layout()

# Lists to store history for plotting
time_history = []
x_history = [[] for _ in range(6)]  # One list for each component

x_old = x0
a_old = np.zeros(6)

t = 0
for jj in range(len(t_eval)):
    x_new, v_new, a_new = velocity_verlet_step(
        x_old,
        a_old,
        dt,
        t,
        dynamics,
        wheels,
        rails,
        contact_eqs,
        material_model,
        fastsim_patch,
        kalker_tables,
        mass_vector,
        rail_y_interpolator,
    )
    t += dt
    x_old = np.concatenate([x_new, v_new])
    # x_old = np.stack((x_new, v_new), axis=0).flatten()
    a_old = a_new

    # Store data for plotting
    time_history.append(t_eval[jj])
    for i in range(6):
        x_history[i].append(x_new[i])

    # Update each subplot
    # for i in range(6):
    #     lines[i].set_data(time_history, x_history[i])

    #     # Adjust axes limits dynamically
    #     axes[i].set_xlim(0, max(time_history) * 1.1)
    #     if len(x_history[i]) > 0:
    #         y_min, y_max = min(x_history[i]), max(x_history[i])
    #         y_range = y_max - y_min if y_max != y_min else 1
    #         axes[i].set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # # Redraw
    # fig.canvas.draw()
    # fig.canvas.flush_events()

    print(f"Time: {t_eval[jj]:.4f} s")

    # Optional: slow down visualization
    # time.sleep(0.01)

# plt.ioff()  # Turn off interactive mode
# plt.show()  # Keep final plot open
