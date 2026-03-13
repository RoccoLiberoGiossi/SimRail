from numba import njit
import numpy as np

@njit(cache=True)
def interpolator(x_rail, x_interp, y_interp):
    return np.interp(x_rail, x_interp, y_interp)

@njit(cache=True, fastmath=True)
def compute_fastsim_core(
    xx, yy, q, r, a, b,
    normal_force, friction,
    L_x, L_y, L_phi,
    nu_x, nu_y, phi, v_x,
    discretization
):
    """
    JIT-compiled core FASTSIM algorithm.
    
    Returns
    -------
    p_x, p_y, s_x, s_y, s_out, g_bound : arrays (discretization, discretization)
    F_x, F_y : float
    """

    a_sq = a*a
    b_sq = b*b
    
    # Pre-allocate arrays
    p_x = np.zeros((discretization, discretization))
    p_y = np.zeros((discretization, discretization))
    s_x = np.zeros((discretization, discretization))
    s_y = np.zeros((discretization, discretization))
    s_out = np.zeros((discretization, discretization))
    g_bound = np.zeros((discretization, discretization))
    
    # Pre-compute constants
    g_const = 2 * normal_force * friction / (np.pi * a_sq * a * b)
    L_x_v_q = L_x * v_x / q
    L_y_v_q = L_y * v_x / q
    nu_x_L_x = nu_x / L_x
    nu_y_L_y = nu_y / L_y
    q_r = q * r
    
    F_x = 0.0
    F_y = 0.0
    
    # Loop over y (columns)
    for jj in range(discretization):
        y_val = yy[jj]
        y_val_sq = y_val * y_val
        
        # Check if we're within ellipse bounds
        if abs(y_val) <= b:
            # Compute a_y: semi-axis at this y
            a_y_sq = a_sq * (1.0 - y_val_sq / b_sq)
            a_y = np.sqrt(a_y_sq)
        else:
            a_y = 0.0
            a_y_sq = 0.0
        
        # Skip if a_y too small
        if a_y <= 1e-11:
            continue
        
        # Pre-compute y-dependent terms
        c_x = (nu_x - phi * y_val) * v_x
        phi_y_L_phi = phi * y_val / L_phi
        
        p_1x = 0.0
        p_1y = 0.0
        
        # Loop over x (rows) - backwards
        for kk in range(discretization - 1, -1, -1):
            x_val = xx[kk]
            x_val_sq = x_val * x_val
            
            # Check if point is inside ellipse at this y
            if x_val_sq > a_y_sq:
                continue
            
            # Compute traction bound
            g_bound[kk, jj] = g_const * (a_y_sq - x_val_sq)
            
            # Pre-compute x-dependent terms
            c_y = (nu_y + phi * x_val) * v_x
            phi_x_L_phi = phi * x_val / L_phi
            
            # Linear pressure update
            p_x_lin = p_1x - q * (nu_x_L_x - phi_y_L_phi)
            p_y_lin = p_1y - q * (nu_y_L_y + phi_x_L_phi)
            
            p_x[kk, jj] = p_x_lin
            p_y[kk, jj] = p_y_lin
            
            # Check saturation
            P_sat_sq = p_x_lin * p_x_lin + p_y_lin * p_y_lin
            P_sat = np.sqrt(P_sat_sq)
            
            g_val = g_bound[kk, jj]
            
            if P_sat >= g_val:
                # Saturated - apply limit
                scale = g_val / P_sat
                p_x[kk, jj] = scale * p_x_lin
                p_y[kk, jj] = scale * p_y_lin
                
                # Compute slip
                s_x[kk, jj] = c_x + L_x_v_q * (p_x[kk, jj] - p_1x)
                s_y[kk, jj] = c_y + L_y_v_q * (p_y[kk, jj] - p_1y)
                
                # Slip magnitude
                s_out[kk, jj] = np.sqrt(s_x[kk, jj]**2 + s_y[kk, jj]**2) / v_x * q
            
            # Update previous values
            p_1x = p_x[kk, jj]
            p_1y = p_y[kk, jj]
            
            # Accumulate forces
            F_x += p_1x
            F_y += p_1y
    
    # Scale forces by grid spacing
    F_x *= q_r
    F_y *= q_r

    # print(xx.shape, yy.shape, p_x.shape, p_y.shape, s_x.shape, s_y.shape, s_out.shape, g_bound.shape)
    
    return p_x, p_y, s_x, s_y, s_out, g_bound, F_x, F_y