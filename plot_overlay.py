import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = "xy_data.csv"

# fitted values from fit_curve.py
THETA_DEG = 30.0
M = 0.03
X = 55.0


def curve_xy(theta_deg, M, X, n=4000):
    theta = np.deg2rad(theta_deg)
    t = np.linspace(6, 60, n)
    wobble = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * np.cos(theta) - wobble * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + wobble * np.cos(theta)
    return x, y


def main():
    df = pd.read_csv(DATA_FILE)
    x_curve, y_curve = curve_xy(THETA_DEG, M, X)

    plt.figure(figsize=(8, 6))
    plt.scatter(df["x"], df["y"], s=6, alpha=0.35, color="tab:gray", label="xy_data.csv (n=1500)")
    plt.plot(x_curve, y_curve, color="tab:red", linewidth=1.6,
              label=f"fit: theta={THETA_DEG} deg, M={M}, X={X}")
    plt.title("fitted curve vs given data points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.savefig("fit_overlay.png", dpi=160)
    print("saved fit_overlay.png")


if __name__ == "__main__":
    main()
