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
from tqdm import tqdm


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

    return F_local
    # return R_wheel @ F_local


def dynamics(
    t,
    x,
    wheels: list[wheel],
    rails: list[rail],
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

    y_rail = rail_y(q[0])

    force_vector_series = []
    slip_series = []
    creep_series = []

    for index, wheel in enumerate(wheels):

        wheel.set_dynamic_state(new_state)
        wheel.calculate_position()

        rails[index].calculate_position(y_rail, 0)

        contact_eq_s[index].patch_search(wheel, rails[index], material_model, discretization=10)

        for _, contacts in enumerate(contact_eq_s[index].contact_patches):
            contact_radius = contacts.Rlocal
            delta_r = wheel.r0 - contact_radius
            if type(contacts.centroid_wheel) != int:
                
                centroid = contacts.centroid_wheel.item()
                
                tan_gamma = np.tan(contacts.contact_angle)
                cos_gamma = np.cos(contacts.contact_angle)
                sin_gamma = np.sin(contacts.contact_angle)
                cos_yaw = np.cos(q[4])
                sin_yaw = np.sin(q[4])
                # v_inv = 1 / v[0] if v[0] != 0 else 0
                v_forward = v[0] if np.abs(v[0]) > 1e-6 else 1e-6 # Avoid hard zero
                v_inv = 1.0 / v_forward

                if wheel.lr == RIGHT:

                    long_disp = contact_radius*tan_gamma*q[4]
                    v_x = v[0] - np.abs(centroid)*v[4] + contact_radius*v[3]
                    v_y = v[1] + contact_radius*v[3]*q[4] - contact_radius*v[5]
                    v_z = v[2] + np.abs(centroid)*v[5] - long_disp*v[3]

                    nu_x = v_inv * (v_x * cos_yaw + v_y * sin_yaw)
                    nu_y = v_inv * (v_y * cos_gamma - v_z * sin_gamma)
                    phi = v_inv * (v[4] * cos_gamma + v[3] * sin_gamma)

                    # nu_x = v_inv * (v[3]*contact_radius - v[0] - np.abs(centroid)*v[4] * cos_gamma)
                    # nu_y = v_inv * v[1] * cos_gamma + q[4] * cos_gamma + v_inv*v[3]*contact_radius*sin_gamma
                    # phi = - v_inv * (v[4] * sin_gamma + v[3] * sin_gamma) * cos_gamma

                elif wheel.lr == LEFT:

                    long_disp = -contact_radius*tan_gamma*q[4]
                    v_x = v[0] + np.abs(centroid)*v[4] + contact_radius*v[3]
                    v_y = v[1] + contact_radius*v[3]*q[4] + contact_radius*v[5]
                    v_z = v[2] - np.abs(centroid)*v[5] - long_disp*v[3]

                    nu_x = v_inv * (v_x * cos_yaw + v_y * sin_yaw)
                    nu_y = v_inv * (v_y * cos_gamma + v_z * sin_gamma)
                    phi = v_inv * (v[4] * cos_gamma - v[3] * sin_gamma)

                    # nu_x = v_inv * (v[3]*contact_radius + v[0] - np.abs(centroid)*v[4] * cos_gamma)
                    # nu_y = v_inv * v[1] * cos_gamma + q[4] * cos_gamma - v_inv*v[3]*contact_radius*sin_gamma
                    # phi = v_inv * (v[4] * sin_gamma + v[3] * sin_gamma) * cos_gamma
                
                # if wheel.lr == RIGHT:

                #     long_disp = contact_radius*tan_gamma*q[4]
                #     v_x = v[0] - np.abs(centroid)*v[4] + contact_radius*(-v[0]/wheel.r0 + v[3])
                #     v_y = v[1] + contact_radius*(-v[0]/wheel.r0 + v[3])*q[4] - contact_radius*v[5]
                #     v_z = v[2] + np.abs(centroid)*v[5] - long_disp*(-v[0]/wheel.r0 + v[3])

                #     nu_x = v_inv * (v_x * cos_yaw + v_y * sin_yaw)
                #     nu_y = v_inv * (v_y * cos_gamma - v_z * sin_gamma)
                #     phi = v_inv * (v[4] * cos_gamma + (-v[0]/wheel.r0 + v[3]) * sin_gamma)

                # elif wheel.lr == LEFT:

                #     long_disp = -contact_radius*tan_gamma*q[4]
                #     v_x = v[0] + np.abs(centroid)*v[4] + contact_radius*(-v[0]/wheel.r0 + v[3])
                #     v_y = v[1] + contact_radius*(-v[0]/wheel.r0 + v[3])*q[4] + contact_radius*v[5]
                #     v_z = v[2] - np.abs(centroid)*v[5] - long_disp*(-v[0]/wheel.r0 + v[3])

                #     nu_x = v_inv * (v_x * cos_yaw + v_y * sin_yaw)
                #     nu_y = v_inv * (v_y * cos_gamma + v_z * sin_gamma)
                #     phi = v_inv * (v[4] * cos_gamma - (-v[0]/wheel.r0 + v[3]) * sin_gamma)

                # print(f"nu_x: {nu_x}, nu_y: {nu_y}, phi: {phi}, cos_gamma: {cos_gamma}, sin_gamma: {sin_gamma}, cos_yaw: {cos_yaw}, sin_yaw: {sin_yaw}")
                # print(f"v_x: {v_x}, v_y: {v_y}, v_z: {v_z}")
                # print(f"v_inv: {v_inv}, v_forward: {v_forward}, cose1: {np.abs(centroid)*v[4]}, cose2: {contact_radius*(-v[0]/wheel.r0 + v[3])}")
                # input()

                creep = tangent_creepage(nu_x=nu_x, nu_y=nu_y, phi=phi)

                tangent_method.calc_tangent_force(
                    contacts, material_model, kalker_tables, creep, new_state
                )

                # print(tangent_method.F_x, tangent_method.F_y, contacts.centroid_wheel.item(), wheel.lr)
                # input()

                force_vector = contact_local_to_global(
                    tangent_method.F_x,
                    tangent_method.F_y,
                    contacts.Q_force,
                    contacts.Y_force,
                    q[4],
                    q[5],
                    contacts.contact_angle,
                )

                force_vector_series.append(np.concatenate([force_vector, [contact_radius * force_vector[0], centroid * force_vector[0], centroid * force_vector[2]]]))
                slip_series.append(np.array([nu_x, nu_y, phi]))
                creep_series.append(np.array([tangent_method.F_x, tangent_method.F_y]))

                # print(force_vector)
                # input()

                # Y_forces.append(contacts.Y_force)
                # Q_forces.append(contacts.Q_force)
                # # Fx_forces.append(tangent_method.F_x)
                # # Fy_forces.append(tangent_method.F_y)
                # # moments_pitch.append(contact_radius * tangent_method.F_x)
                # # moments_yaw.append(contacts.centroid_wheel.item() * tangent_method.F_x)
                # # moments_roll.append(contacts.centroid_wheel.item() * contacts.Q_force)
                # creep_xs.append(nu_x)
                # creep_ys.append(nu_y)
                # phi_s.append(phi)
                # delta_rs.append(delta_r)
                # centroids.append(contacts.centroid_wheel.item())

                F_vector[0] -= force_vector[0]
                F_vector[1] += force_vector[1]
                F_vector[2] += force_vector[2]
                F_vector[3] += contact_radius * force_vector[0]
                F_vector[4] -= centroid * force_vector[0]
                F_vector[5] += centroid * force_vector[2]

            else:
                force_vector_series.append(np.zeros(6))
                slip_series.append(np.zeros(3))
                creep_series.append(np.zeros(2))

    # Y_forces_series.append(Y_forces)
    # Q_forces_series.append(Q_forces)
    # rail_position.append(y_rail)
    # Wheelset_y_position.append(new_state.state_y)
    # Wheelset_z_position.append(new_state.state_z)

    F_vector[2] -= wheelset_mass[0] * 9.81
    # print(f"Total Force Vector: {F_vector}")
    dx[6:12] = -F_vector / wheelset_mass

    # print(f"doing...{t}")

    # global _fig, _ax_1, _ax_2, _ax_3, _ax_4, _ax_5, _ax_6

    # # Create the figure only once
    # if _fig is None:
    #     # _fig, _ax_1 = plt.subplots(1, 1, figsize=(10, 4))
    #     _fig, axes = plt.subplots(2, 3, figsize=(10, 4))
    #     _ax_1, _ax_2, _ax_3, _ax_4, _ax_5, _ax_6 = axes.flatten()
    #     _ax_1.set_title(f"Wheel {t} s")
    #     _ax_1.set_aspect("equal")
    #     _ax_2.set_title("Y Forces")
    #     _ax_3.set_title("Q Forces")
    #     _ax_4.set_title("Rail Position")
    #     _ax_5.set_title("Wheelset y Position")
    #     _ax_6.set_title("Rail Position")
    #     # _ax_2.set_aspect("equal")
    #     plt.show()

    # # Clear the old plots
    # _ax_1.cla()
    # _ax_2.cla()
    # _ax_3.cla()
    # _ax_4.cla()
    # _ax_5.cla()
    # _ax_6.cla()

    # # Plot wheel geometry
    # _ax_1.plot(rails[1].rail_profile_pos[:, 0], rails[1].rail_profile_pos[:, 1], c="r")
    # _ax_1.plot(wheels[1].wheel_profile_pos[:, 0], wheels[1].wheel_profile_pos[:, 1], c="black")
    # _ax_1.plot(rails[0].rail_profile_pos[:, 0], rails[0].rail_profile_pos[:, 1], c="r")
    # _ax_1.plot(wheels[0].wheel_profile_pos[:, 0], wheels[0].wheel_profile_pos[:, 1], c="black")
    # _ax_1.set_title(f"Wheel {t} s")
    # _ax_1.set_aspect("equal")

    # # Plot rail geometry (select correct rail)
    # _ax_2.plot(Y_forces_series)
    # _ax_2.grid(True)
    # _ax_2.set_title("Y Forces")
    # # _ax_2.set_aspect("equal")

    # # Plot Q forces
    # _ax_3.plot(Q_forces_series)
    # _ax_3.grid(True)
    # _ax_3.set_title("Q Forces")
    
    # # Plot rail pos
    # _ax_4.plot(rail_position)
    # _ax_4.grid(True)
    # _ax_4.set_title("Rail Position")
    
    # # Plot rail pos
    # _ax_5.plot(np.array(Wheelset_y_position)-np.array(rail_position))
    # _ax_5.grid(True)
    # _ax_5.set_title("Wheelset y Position")

    # # Plot rail pos
    # _ax_6.plot(Wheelset_z_position)
    # _ax_6.grid(True)
    # _ax_6.set_title("Wheelset z Position")
    # # _ax_3.set_aspect("equal")

    # # Refresh only
    # _fig.canvas.draw()
    # _fig.canvas.flush_events()

    return dx, F_vector, force_vector_series, slip_series, creep_series


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
    a_star, Force, force_vector_series, slip_series, creep_series = force_func(
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
    return x_new_star, v_new, a_new, Force, force_vector_series, slip_series, creep_series

wheels = [wheel(), wheel(lr=LEFT)]
rails = [rail(rail_inclination=20), rail(lr=LEFT, rail_inclination=20)]

n_patches = 3
contact_eqs = [
    NormalContactSolver(eqv_el_normal_contact(), n_patches),
    NormalContactSolver(eqv_el_normal_contact(), n_patches),
]

material_model = material()

kalker_tables = kalker_coefficients(material_properties=material_model)
fastsim_patch = fastsim(discratization=10)

wheelset_mass = 1200.0
I_x = 110.0
I_y = 800.0
I_z = 800.0

patches_per_deltays_eq, result_state_pressure_eq = static_contact(
    [0],
    wheels[0],
    rails[0],
    contact_eqs[0],
    material_model,
    Q_solve=wheelset_mass * 9.81 / 2,
    discretization=10
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
x0[6] = 50 / 3.6
x0[9] = -x0[6]/wheels[0].r0

mass_vector = np.array([wheelset_mass, wheelset_mass, wheelset_mass, I_x, I_y, I_z])

d_max = 20 # [m]

tspan = (0, d_max / x0[6])
dt = 0.5e-3

print(f"Total simulation time: {tspan[1]} s with time step: {dt} s, with vehicle speed {x0[6]*3.6} km/h")

t_eval = np.arange(tspan[0], tspan[1] + dt, dt)

d_impulse = 0.1e-3  # max displacement
ds = 1e-5
d_eval = np.arange(0, d_max + ds, ds)

ramp_start = 1
ramp_end = 1.25
start_index = int(ramp_start / ds)
end_index = int(ramp_end / ds)

ramp_length = end_index - start_index

rail_y = np.zeros_like(d_eval)
# rail_y[start_index:end_index] = d_impulse * np.linspace(0, 1, ramp_length)
# rail_y[end_index:end_index+ramp_length*2] = - d_impulse * np.linspace(0, 2, ramp_length*2) + d_impulse
# rail_y[end_index+ramp_length*2:end_index+ramp_length*3] = d_impulse * np.linspace(0, 1, ramp_length) - d_impulse
# rail_y[ramp_end*2:] = 0,0

rail_y_interpolator = PchipInterpolator(d_eval, rail_y)
fig, ax = plt.subplots()
ax.plot(d_eval, rail_y)
plt.show()

input()

# plt.ion()

# _fig = None
# _ax_1 = None
# _ax_2 = None
# _ax_3 = None
# _ax_4 = None
# _ax_5 = None
# _ax_6 = None

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

# Lists to store history for plotting
time_history = []
x_history = [[] for _ in range(6)]  # One list for each component
v_history = [[] for _ in range(6)]  # One list for each component
a_history = [[] for _ in range(6)]  # One list for each component
Force_history = [[] for _ in range(6)]  # One list for each component
force_vector_series_history = []  # To store force vectors for each time step
slip_series_history = []  # To store slip values for each time step
creep_series_history = []  # To store creep values for each time step

x_old = x0
a_old = np.zeros(6)

t = 0
for jj in tqdm(range(len(t_eval))):
    x_new, v_new, a_new, Force, force_vector_series, slip_series, creep_series = velocity_verlet_step(
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
    # print(force_vector_series)
    # input()
    t += dt
    x_old = np.concatenate([x_new, v_new])
    # x_old = np.stack((x_new, v_new), axis=0).flatten()
    a_old = a_new

    # Store data for plotting
    time_history.append(t_eval[jj])
    for i in range(6):
        x_history[i].append(x_new[i])
        v_history[i].append(v_new[i])
        a_history[i].append(a_new[i])
        Force_history[i].append(Force[i])
    force_vector_series_history.append(force_vector_series)
    slip_series_history.append(slip_series)
    creep_series_history.append(creep_series)
force_vector_series_history = np.array(force_vector_series_history)
slip_series_history = np.array(slip_series_history)
creep_series_history = np.array(creep_series_history)
print(force_vector_series_history.shape)
print(slip_series_history.shape)
input()

fig, axes = plt.subplots(1, 3, figsize=(12, 8))
axes[0].plot(time_history, force_vector_series_history[:, 0, 0], linewidth=2)
axes[0].plot(time_history, force_vector_series_history[:, 1, 0], linewidth=2)
axes[0].plot(time_history, force_vector_series_history[:, 2, 0], linewidth=2)
axes[0].plot(time_history, force_vector_series_history[:, 3, 0], linewidth=2)
axes[0].plot(time_history, force_vector_series_history[:, 4, 0], linewidth=2)
axes[0].plot(time_history, force_vector_series_history[:, 5, 0], linewidth=2)
axes[1].plot(time_history, force_vector_series_history[:, 0, 1], linewidth=2)
axes[1].plot(time_history, force_vector_series_history[:, 1, 1], linewidth=2)
axes[1].plot(time_history, force_vector_series_history[:, 2, 1], linewidth=2)
axes[1].plot(time_history, force_vector_series_history[:, 3, 1], linewidth=2)
axes[1].plot(time_history, force_vector_series_history[:, 4, 1], linewidth=2)
axes[1].plot(time_history, force_vector_series_history[:, 5, 1], linewidth=2)
axes[2].plot(time_history, force_vector_series_history[:, 0, 2], linewidth=2)
axes[2].plot(time_history, force_vector_series_history[:, 1, 2], linewidth=2)
axes[2].plot(time_history, force_vector_series_history[:, 2, 2], linewidth=2)
axes[2].plot(time_history, force_vector_series_history[:, 3, 2], linewidth=2)
axes[2].plot(time_history, force_vector_series_history[:, 4, 2], linewidth=2)
axes[2].plot(time_history, force_vector_series_history[:, 5, 2], linewidth=2)
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 8))
axes[0].plot(time_history, force_vector_series_history[:, 0, 3], linewidth=2)
axes[0].plot(time_history, force_vector_series_history[:, 1, 3], linewidth=2)
axes[0].plot(time_history, force_vector_series_history[:, 2, 3], linewidth=2)
axes[0].plot(time_history, force_vector_series_history[:, 3, 3], linewidth=2)
axes[0].plot(time_history, force_vector_series_history[:, 4, 3], linewidth=2)
axes[0].plot(time_history, force_vector_series_history[:, 5, 3], linewidth=2)
axes[1].plot(time_history, force_vector_series_history[:, 0, 4], linewidth=2)
axes[1].plot(time_history, force_vector_series_history[:, 1, 4], linewidth=2)
axes[1].plot(time_history, force_vector_series_history[:, 2, 4], linewidth=2)
axes[1].plot(time_history, force_vector_series_history[:, 3, 4], linewidth=2)
axes[1].plot(time_history, force_vector_series_history[:, 4, 4], linewidth=2)
axes[1].plot(time_history, force_vector_series_history[:, 5, 4], linewidth=2)
axes[2].plot(time_history, force_vector_series_history[:, 0, 5], linewidth=2)
axes[2].plot(time_history, force_vector_series_history[:, 1, 5], linewidth=2)
axes[2].plot(time_history, force_vector_series_history[:, 2, 5], linewidth=2)
axes[2].plot(time_history, force_vector_series_history[:, 3, 5], linewidth=2)
axes[2].plot(time_history, force_vector_series_history[:, 4, 5], linewidth=2)
axes[2].plot(time_history, force_vector_series_history[:, 5, 5], linewidth=2)
fig.tight_layout()
plt.show()
    
fig, axes = plt.subplots(1, 3, figsize=(12, 8))
axes[0].plot(time_history, slip_series_history[:, 0, 0], linewidth=2)
axes[0].plot(time_history, slip_series_history[:, 1, 0], linewidth=2)
axes[0].plot(time_history, slip_series_history[:, 2, 0], linewidth=2)
axes[0].plot(time_history, slip_series_history[:, 3, 0], linewidth=2)
axes[0].plot(time_history, slip_series_history[:, 4, 0], linewidth=2)
axes[0].plot(time_history, slip_series_history[:, 5, 0], linewidth=2)
axes[1].plot(time_history, slip_series_history[:, 0, 1], linewidth=2)
axes[1].plot(time_history, slip_series_history[:, 1, 1], linewidth=2)
axes[1].plot(time_history, slip_series_history[:, 2, 1], linewidth=2)
axes[1].plot(time_history, slip_series_history[:, 3, 1], linewidth=2)
axes[1].plot(time_history, slip_series_history[:, 4, 1], linewidth=2)
axes[1].plot(time_history, slip_series_history[:, 5, 1], linewidth=2)
axes[2].plot(time_history, slip_series_history[:, 0, 2], linewidth=2)
axes[2].plot(time_history, slip_series_history[:, 1, 2], linewidth=2)
axes[2].plot(time_history, slip_series_history[:, 2, 2], linewidth=2)
axes[2].plot(time_history, slip_series_history[:, 3, 2], linewidth=2)
axes[2].plot(time_history, slip_series_history[:, 4, 2], linewidth=2)
axes[2].plot(time_history, slip_series_history[:, 5, 2], linewidth=2)
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes[0].plot(time_history, creep_series_history[:, 0, 0], linewidth=2)
axes[0].plot(time_history, creep_series_history[:, 1, 0], linewidth=2)
axes[0].plot(time_history, creep_series_history[:, 2, 0], linewidth=2)
axes[0].plot(time_history, creep_series_history[:, 3, 0], linewidth=2)
axes[0].plot(time_history, creep_series_history[:, 4, 0], linewidth=2)
axes[0].plot(time_history, creep_series_history[:, 5, 0], linewidth=2)
axes[1].plot(time_history, creep_series_history[:, 0, 1], linewidth=2)
axes[1].plot(time_history, creep_series_history[:, 1, 1], linewidth=2)
axes[1].plot(time_history, creep_series_history[:, 2, 1], linewidth=2)
axes[1].plot(time_history, creep_series_history[:, 3, 1], linewidth=2)
axes[1].plot(time_history, creep_series_history[:, 4, 1], linewidth=2)
axes[1].plot(time_history, creep_series_history[:, 5, 1], linewidth=2)
fig.tight_layout()
plt.show()
    

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

    # print(f"Time: {t_eval[jj]:.4f} s")

    # Optional: slow down visualization
    # time.sleep(0.01)

fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()  # Flatten to 1D array for easy indexing

names = ["x", "y", "z", "pitch", "yaw", "roll"]
# Initialize empty lines for each subplot
for i, ax in enumerate(axes):
    ax.plot(time_history, x_history[i], "b-", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"x[{i}]")
    ax.set_title(f"State Variable {names[i]}")
    ax.grid(True)

fig.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()  # Flatten to 1D array for easy indexing

names = ["x", "y", "z", "pitch", "yaw", "roll"]
# Initialize empty lines for each subplot
for i, ax in enumerate(axes):
    ax.plot(time_history, v_history[i], "b-", linewidth=2)
    if names[i] in ["pitch"]:
        ax.plot(time_history, -1*np.array(v_history[0])/wheels[0].r0, "r-.", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"v[{i}]")
    ax.set_title(f"State Velocity {names[i]}")
    ax.grid(True)

fig.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()  # Flatten to 1D array for easy indexing

names = ["x", "y", "z", "pitch", "yaw", "roll"]
# Initialize empty lines for each subplot
for i, ax in enumerate(axes):
    ax.plot(time_history, a_history[i], "b-", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"a[{i}]")
    ax.set_title(f"State Acceleration {names[i]}")
    ax.grid(True)

fig.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()  # Flatten to 1D array for easy indexing

names = ["x", "y", "z", "pitch", "yaw", "roll"]
# Initialize empty lines for each subplot
for i, ax in enumerate(axes):
    ax.plot(time_history, Force_history[i], "b-", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Force[{i}]")
    ax.set_title(f"Force Variable {names[i]}")
    ax.grid(True)

fig.tight_layout()
plt.show()

# plt.ioff()  # Turn off interactive mode
# plt.show()  # Keep final plot open
