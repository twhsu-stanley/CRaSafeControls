import numpy as np
import sympy as sp

def sindy_prediction_symbolic(x, u, feature_names, coefficients, feature_indices = None):
    """ Compute the SINDy model predciction using symbolic expressions"""
    # x: symbolic array (n_states-by-1)
    # u: symbolic array (n_controls-by-1)
    # feature_names: list (len = n_features)
    # coefficients: array (size = n_states x n_features)

    if feature_indices is None:
        n_features = len(feature_names)
        feature_indices = range(n_features)
    
    n_states = coefficients.shape[0]
    n_controls = u.shape[0]

    for s in range(n_states):
        locals()[f'x{s}'] = x[s]

    for c in range(n_controls):
        locals()[f'u{c}'] = u[c]

    f = []
    for s in range(n_states):
        fs = 0
        for i in feature_indices:
            fs = fs + eval(feature_names[i]) * coefficients[s,i]
        f.append(fs)
    f = sp.Matrix(f)
    
    return f

# Functions for adaptive controls
def phi(a_hat, a_hat_norm_max, epsilon):
    """Compute the barrier function φ(â)."""
    return (np.linalg.norm(a_hat,2)**2 - a_hat_norm_max**2) / (2 * epsilon * a_hat_norm_max + epsilon**2)

def grad_phi(a_hat, a_hat_norm_max, epsilon):
    """Compute the gradient ∇φ(â) = 2 * â / (2εa_hat_norm_max + ε²)."""
    return (2 * a_hat) / (2 * epsilon * a_hat_norm_max + epsilon**2)

def projection_operator(a_hat, y, a_hat_norm_max, epsilon):
    """
    Implements the adaptive control projection operator:
    Proj(a_hat, y, φ)

    Parameters:
    - a_hat: current parameter estimate (np.ndarray)
    - y: nominal adaptation signal (np.ndarray)
    - a_hat_norm_max: upper bound on norm of a_hat
    - epsilon: small positive scalar (for soft boundary enforcement)

    Returns:
    - projected update (np.ndarray)
    """
    phi_val = phi(a_hat, a_hat_norm_max, epsilon)
    grad_phi_val = grad_phi(a_hat, a_hat_norm_max, epsilon)

    if phi_val > 0 and (y.T @ grad_phi_val).item() > 0:
        projection_matrix = (grad_phi_val @ grad_phi_val.T) / np.linalg.norm(grad_phi_val,2)**2
        #projection_matrix = np.outer(grad_phi_val, grad_phi_val) / np.dot(grad_phi_val, grad_phi_val)
        correction = projection_matrix @ y * phi_val
        return y - correction
    else:
        return y

def wrapToPi(angle):
    """Wrap angle to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi