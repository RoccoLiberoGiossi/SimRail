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
from matplotlib.animation import FuncAnimation
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

    return R_wheel @ F_local


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
    variable_series = []

    # wheels = [wheel(), wheel(lr=LEFT)]
    rails = [rail(rail_inclination=20), rail(lr=LEFT, rail_inclination=20)]

    for index, wheel_comp in enumerate(wheels):

        wheel_comp.set_dynamic_state(new_state)
        wheel_comp.calculate_position()

        rails[index].calculate_position(y_rail, 0)

        contact_eq_s[index].patch_search(wheel_comp, rails[index], material_model, discretization=58)

        for contact_number, contacts in enumerate(contact_eq_s[index].contact_patches):
            contact_radius = contacts.Rlocal
            delta_r = wheel_comp.r0 - contact_radius
            if type(contacts.centroid_wheel) != int:
                
                centroid = contacts.centroid_wheel.item()
                
                tan_gamma = np.tan(contacts.contact_angle)
                cos_gamma = np.cos(contacts.contact_angle)
                sin_gamma = np.sin(contacts.contact_angle)
                cos_yaw = np.cos(q[4])
                sin_yaw = np.sin(q[4])
                v_forward = v[0] if np.abs(v[0]) > 1e-6 else 1e-6 # Avoid hard zero
                v_inv = 1.0 / v_forward

                nu_x, nu_y, phi = 0.0, 0.0, 0.0

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

                creep = tangent_creepage(nu_x=nu_x, nu_y=nu_y, phi=phi)

                tangent_method.calc_tangent_force(
                    contacts, material_model, kalker_tables, creep, new_state
                )

                # if np.abs(tangent_method.F_x) > 100e3 or np.abs(tangent_method.F_y) > 100e3 or np.abs(contacts.Q_force) > 100e3 or np.abs(contacts.Y_force) > 100e3:
                #     print("High tangential force detected! Check creep values and contact conditions.")
                #     print(f"Contact patch {contact_number} for wheel {index} has unusually high forces.")
                #     print(f"nu_x: {nu_x}, nu_y: {nu_y}, phi: {phi}")
                #     print(f"Contact radius: {contact_radius}, Centroid: {centroid}")
                #     print(f"Contact angle: {contacts.contact_angle}")
                #     print(f"F_x: {tangent_method.F_x}, F_y: {tangent_method.F_y}")
                #     print(f"Wheel velocity: {v_forward}, v_x: {v_x}, v_y: {v_y}, v_z: {v_z}")
                #     print(f"Wheel state: {new_state}")
                #     print(f"Wheel a: {contacts.semi_axis_a}, b: {contacts.semi_axis_b}")
                #     print(f"Wheel Q: {contacts.Q_force}, Y: {contacts.Y_force}")
                #     input()

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
                variable_series.append(np.array([contact_radius, contacts.contact_angle]))

                F_vector[0] += force_vector[0]
                F_vector[1] += force_vector[1]
                F_vector[2] += force_vector[2]
                F_vector[3] -= contact_radius * force_vector[0]
                F_vector[4] -= centroid * force_vector[0]
                F_vector[5] += centroid * force_vector[2]

            else:
                force_vector_series.append(np.zeros(6))
                slip_series.append(np.zeros(3))
                creep_series.append(np.zeros(2))
                variable_series.append(np.zeros(2))

    F_vector[2] -= wheelset_mass[0] * 9.81
    dx[6:12] = -F_vector / wheelset_mass

    return dx#, F_vector, force_vector_series, slip_series, creep_series, variable_series


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
    a_star, Force, force_vector_series, slip_series, creep_series, variable_series = force_func(
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
    return x_new_star, v_new, a_new, Force, force_vector_series, slip_series, creep_series, variable_series

wheels = [wheel(), wheel(lr=LEFT)]
rails = [rail(rail_inclination=20), rail(lr=LEFT, rail_inclination=20)]

n_patches = 3
contact_eqs = [
    NormalContactSolver(eqv_el_normal_contact(), n_patches),
    NormalContactSolver(eqv_el_normal_contact(), n_patches),
]

material_model = material()

kalker_tables = kalker_coefficients(material_properties=material_model)
fastsim_patch = fastsim(discratization=58)

wheelset_mass = 1200.0
I_x = 110.0
I_y = 800.0
I_z = 800.0

patches_per_deltays_eq, result_state_pressure_eq = static_contact(
    np.array([0]),
    wheels[0],
    rails[0],
    contact_eqs[0],
    material_model,
    Q_solve=wheelset_mass * 9.81 / 2,
    discretization=58
)

print(patches_per_deltays_eq["patch_0"][0][0])

patches_per_deltays_eq, result_state_pressure_eq = static_contact(
    np.array([0]),
    wheels[1],
    rails[1],
    contact_eqs[1],
    material_model,
    Q_solve=wheelset_mass * 9.81 / 2,
)

print(patches_per_deltays_eq["patch_0"][0][0])
input()

x0 = np.zeros(12)
x0[2] = patches_per_deltays_eq["patch_0"][0][0]
x0[6] = 20 / 3.6
x0[9] = -x0[6]/wheels[0].r0

mass_vector = np.array([wheelset_mass, wheelset_mass, wheelset_mass, I_x, I_y, I_z])

d_max = 15 # [m]

tspan = (0, d_max / x0[6])
dt = 0.5e-3

print(f"Total simulation time: {tspan[1]} s with time step: {dt} s, with vehicle speed {x0[6]*3.6} km/h")

t_eval = np.linspace(tspan[0], tspan[1], int(tspan[1] / dt))

d_impulse = 10e-3  # max displacement
ds = 1e-5
d_eval = np.arange(0, d_max + ds, ds)

ramp_start = 1
ramp_end = 2
start_index = int(ramp_start / ds)
end_index = int(ramp_end / ds)

ramp_length = end_index - start_index

rail_y = np.zeros_like(d_eval)

rail_y[start_index:end_index] = d_impulse * 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, ramp_length)))
rail_y[end_index:] = d_impulse
# rail_y[end_index:end_index + ramp_length * 2] = d_impulse * np.cos(np.pi * np.linspace(0, 1, ramp_length * 2))
# rail_y[end_index + ramp_length * 2:end_index + ramp_length * 3] = -d_impulse * 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, ramp_length)))

rail_y_interpolator = PchipInterpolator(d_eval, rail_y)
d_eval_prova = np.arange(0, d_max + 1e-3, 1e-3)
prova_rail = rail_y_interpolator(d_eval_prova)
fig, ax = plt.subplots()
ax.plot(d_eval, rail_y)
ax.plot(d_eval_prova, prova_rail, "--")
plt.show()

fig, ax = plt.subplots()
rails[0].calculate_position(rail_y[0], 0)
ax.plot(rails[0].rail_profile_pos[:, 0], rails[0].rail_profile_pos[:, 1])
rails[0].calculate_position(rail_y[end_index - 1], 0)
ax.plot(rails[0].rail_profile_pos[:, 0], rails[0].rail_profile_pos[:, 1])
rails[0].calculate_position(rail_y[-1], 0)
ax.plot(rails[0].rail_profile_pos[:, 0], rails[0].rail_profile_pos[:, 1])
plt.show()

fig, ax = plt.subplots()
rails[1].calculate_position(rail_y[0], 0)
ax.plot(rails[1].rail_profile_pos[:, 0], rails[1].rail_profile_pos[:, 1])
rails[1].calculate_position(rail_y[end_index - 1], 0)
ax.plot(rails[1].rail_profile_pos[:, 0], rails[1].rail_profile_pos[:, 1])
rails[1].calculate_position(rail_y[-1], 0)
ax.plot(rails[1].rail_profile_pos[:, 0], rails[1].rail_profile_pos[:, 1])
plt.show()

input()

with tqdm(total=tspan[1], unit="s", desc="Simulation Progress") as pbar:
    last_t = [0]

    def dynamics_with_progress(t, x, *args):
        # Update progress bar only when t increases
        if t > last_t[0]:
            pbar.update(t - last_t[0])
            last_t[0] = t
        return dynamics(t, x, *args)

    sol = solve_ivp(
        dynamics_with_progress,
        tspan,
        x0,
        t_eval=t_eval,
        method="RK45",
        max_step=dt,
        args=(
            wheels,
            rails,
            contact_eqs,
            material_model,
            fastsim_patch,
            kalker_tables,
            mass_vector,
            rail_y_interpolator,
        ),  # pass extra parameters
    )

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.plot(sol.t, sol.y[i, :])

fig.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.plot(sol.t, sol.y[i+6, :])

fig.tight_layout()
plt.show()

fig, ax = plt.subplots(ncols=2, figsize=(12, 6))

def update(i):
    ax[0].cla()  # clear axes each frame
    ax[1].cla()  # clear axes each frame

    wheels = [wheel(), wheel(lr=LEFT)]
    rails = [rail(rail_inclination=20), rail(lr=LEFT, rail_inclination=20)]
    
    t = sol.t[i]
    x = sol.y[:, i]

    new_state = dynamic_state(
        state_x=x[0], state_y=x[1], state_z=x[2],
        state_pitch=x[3], state_yaw=x[4], state_roll=x[5],
        state_v_x=x[6], state_v_y=x[7], state_v_z=x[8],
        state_v_pitch=x[9], state_v_yaw=x[10], state_v_roll=x[11],
    )

    y_rail = rail_y_interpolator(x[0])

    for index, wheel_x in enumerate(wheels):
        wheel_x.set_dynamic_state(new_state)
        wheel_x.calculate_position()
        rails[index].calculate_position(y_rail, 0)

    ax[0].plot(rails[1].rail_profile_pos[:, 0], rails[1].rail_profile_pos[:, 1], c="r")
    ax[0].plot(wheels[1].wheel_profile_pos[:, 0], wheels[1].wheel_profile_pos[:, 1], c="black")
    ax[1].plot(rails[0].rail_profile_pos[:, 0], rails[0].rail_profile_pos[:, 1], c="r")
    ax[1].plot(wheels[0].wheel_profile_pos[:, 0], wheels[0].wheel_profile_pos[:, 1], c="black")
    ax[0].set_title(f"t = {t:.4f} s")
    ax[1].set_title(f"t = {t:.4f} s")
    ax[0].set_aspect("equal")
    ax[1].set_aspect("equal")

frames = range(0, len(sol.t), 100)
ani = FuncAnimation(fig, update, frames=frames, interval=50)
plt.show()

# fig, ax = plt.subplots()
# for i in range(0,len(sol.t), 100):

#     print(i)
#     t = sol.t[i]
#     x = sol.y[:, i]

#     new_state = dynamic_state(
#         state_x=x[0],
#         state_y=x[1],
#         state_z=x[2],
#         state_pitch=x[3],
#         state_yaw=x[4],
#         state_roll=x[5],
#         state_v_x=x[6],
#         state_v_y=x[7],
#         state_v_z=x[8],
#         state_v_pitch=x[9],
#         state_v_yaw=x[10],
#         state_v_roll=x[11],
#     )

#     y_rail = rail_y_interpolator(x[0])
#     print(y_rail, x[0])

#     for index, wheel_x in enumerate(wheels):

#         wheel_x.set_dynamic_state(new_state)
#         wheel_x.calculate_position()

#         rails[index].calculate_position(y_rail, 0)

#     # Plot wheel geometry
#     ax.plot(rails[1].rail_profile_pos[:, 0], rails[1].rail_profile_pos[:, 1], c="r")
#     ax.plot(wheels[1].wheel_profile_pos[:, 0], wheels[1].wheel_profile_pos[:, 1], c="black")
#     ax.plot(rails[0].rail_profile_pos[:, 0], rails[0].rail_profile_pos[:, 1], c="r")
#     ax.plot(wheels[0].wheel_profile_pos[:, 0], wheels[0].wheel_profile_pos[:, 1], c="black")
#     ax.set_title(f"Wheel {t} s")
#     ax.set_aspect("equal")

# plt.show()

input()

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
variable_series_history = []  # To store creep values for each time step

x_old = x0
a_old = np.zeros(6)

t = 0
for jj in tqdm(range(len(t_eval))):
    x_new, v_new, a_new, Force, force_vector_series, slip_series, creep_series, variable_series = velocity_verlet_step(
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
    variable_series_history.append(variable_series)
force_vector_series_history = np.array(force_vector_series_history)
slip_series_history = np.array(slip_series_history)
creep_series_history = np.array(creep_series_history)
variable_series_history = np.array(variable_series_history)
print(force_vector_series_history.shape)
print(slip_series_history.shape)
print(variable_series_history.shape)
input()

fig, ax = plt.subplots(1, 2)
ax[0].plot(time_history, variable_series_history[:, 0, 0])
# ax.plot(time_history, variable_series_history[:, 1, 0])
# ax.plot(time_history, variable_series_history[:, 2, 0])
ax[0].plot(time_history, variable_series_history[:, 3, 0])
# ax.plot(time_history, variable_series_history[:, 4, 0])
# ax.plot(time_history, variable_series_history[:, 5, 0])
ax[0].plot(time_history, np.ones_like(time_history)*wheels[0].r0, "r-.")
ax[1].plot(time_history, variable_series_history[:, 0, 1])
# ax.plot(time_history, variable_series_history[:, 1, 0])
# ax.plot(time_history, variable_series_history[:, 2, 0])
ax[1].plot(time_history, variable_series_history[:, 3, 1])
# ax.plot(time_history, variable_series_history[:, 4, 0])
# ax.plot(time_history, variable_series_history[:, 5, 0])
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0,0].plot(time_history, force_vector_series_history[:, 0, 0], linewidth=2)
axes[0,0].plot(time_history, force_vector_series_history[:, 1, 0], linewidth=2)
axes[0,0].plot(time_history, force_vector_series_history[:, 2, 0], linewidth=2)
axes[0,0].plot(time_history, force_vector_series_history[:, 3, 0], linewidth=2)
axes[0,0].plot(time_history, force_vector_series_history[:, 4, 0], linewidth=2)
axes[0,0].plot(time_history, force_vector_series_history[:, 5, 0], linewidth=2)
axes[0,1].plot(time_history, force_vector_series_history[:, 0, 1], linewidth=2)
axes[0,1].plot(time_history, force_vector_series_history[:, 1, 1], linewidth=2)
axes[0,1].plot(time_history, force_vector_series_history[:, 2, 1], linewidth=2)
axes[0,1].plot(time_history, force_vector_series_history[:, 3, 1], linewidth=2)
axes[0,1].plot(time_history, force_vector_series_history[:, 4, 1], linewidth=2)
axes[0,1].plot(time_history, force_vector_series_history[:, 5, 1], linewidth=2)
axes[0,2].plot(time_history, force_vector_series_history[:, 0, 2], linewidth=2)
axes[0,2].plot(time_history, force_vector_series_history[:, 1, 2], linewidth=2)
axes[0,2].plot(time_history, force_vector_series_history[:, 2, 2], linewidth=2)
axes[0,2].plot(time_history, force_vector_series_history[:, 3, 2], linewidth=2)
axes[0,2].plot(time_history, force_vector_series_history[:, 4, 2], linewidth=2)
axes[0,2].plot(time_history, force_vector_series_history[:, 5, 2], linewidth=2)
axes[1,0].plot(time_history, force_vector_series_history[:, 0, 3], linewidth=2)
axes[1,0].plot(time_history, force_vector_series_history[:, 1, 3], linewidth=2)
axes[1,0].plot(time_history, force_vector_series_history[:, 2, 3], linewidth=2)
axes[1,0].plot(time_history, force_vector_series_history[:, 3, 3], linewidth=2)
axes[1,0].plot(time_history, force_vector_series_history[:, 4, 3], linewidth=2)
axes[1,0].plot(time_history, force_vector_series_history[:, 5, 3], linewidth=2)
axes[1,1].plot(time_history, force_vector_series_history[:, 0, 4], linewidth=2)
axes[1,1].plot(time_history, force_vector_series_history[:, 1, 4], linewidth=2)
axes[1,1].plot(time_history, force_vector_series_history[:, 2, 4], linewidth=2)
axes[1,1].plot(time_history, force_vector_series_history[:, 3, 4], linewidth=2)
axes[1,1].plot(time_history, force_vector_series_history[:, 4, 4], linewidth=2)
axes[1,1].plot(time_history, force_vector_series_history[:, 5, 4], linewidth=2)
axes[1,2].plot(time_history, force_vector_series_history[:, 0, 5], linewidth=2)
axes[1,2].plot(time_history, force_vector_series_history[:, 1, 5], linewidth=2)
axes[1,2].plot(time_history, force_vector_series_history[:, 2, 5], linewidth=2)
axes[1,2].plot(time_history, force_vector_series_history[:, 3, 5], linewidth=2)
axes[1,2].plot(time_history, force_vector_series_history[:, 4, 5], linewidth=2)
axes[1,2].plot(time_history, force_vector_series_history[:, 5, 5], linewidth=2)
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
        ax.plot(time_history, -1*np.array(v_history[0])/variable_series_history[:, 0, 0], linewidth=2)
        ax.plot(time_history, -1*np.array(v_history[0])/variable_series_history[:, 3, 0], linewidth=2)
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
