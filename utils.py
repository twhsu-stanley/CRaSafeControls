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

def wrapToPi(angle):
    """Wrap angle to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi