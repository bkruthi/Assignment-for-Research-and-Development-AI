import numpy as np, pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree

df = pd.read_csv('xy_data.csv')
data = df[['x', 'y']].values

def curve_points(theta, M, X, n=3000, tmin=6, tmax=60):
    t = np.linspace(tmin, tmax, n)
    ex = np.exp(M * np.abs(t))
    s = np.sin(0.3 * t)
    x = t*np.cos(theta) - ex*s*np.sin(theta) + X
    y = 42 + t*np.sin(theta) + ex*s*np.cos(theta)
    return np.column_stack([x, y])

def loss(params):
    theta, M, X = params
    tree = cKDTree(curve_points(theta, M, X))
    dist, _ = tree.query(data)
    return np.mean(dist**2)

bounds = [(0, np.deg2rad(50)), (-0.05, 0.05), (0, 100)]
res = differential_evolution(loss, bounds, seed=42, maxiter=200, popsize=30, polish=True)
res = minimize(loss, res.x, method='Nelder-Mead', options={'xatol':1e-9,'fatol':1e-12})
theta, M, X = res.x
print(np.degrees(theta), M, X)  