
"""
Illustrative MOSFET LDD resistance extraction.
The data are synthetic examples, NOT experimental measurements.

Run:
    py -m pip install numpy matplotlib
    py ldd_resistance_extraction.py
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Measurement/model assumptions
VDS = 0.050       # Drain-source voltage: 50 mV
VT = 0.50         # Assumed threshold voltage (V)
R_NiSi = 10.0     # Illustrative NiSi resistance (ohm)
R_SD = 20.0       # Illustrative source/drain resistance per side (ohm)

# Synthetic example model
Rext_true = 200.0  # Expected resistance intercept (ohm)
k_true = 200.0     # Expected slope (ohm*V)

# 2. Gate-voltage sweep
overdrive = np.array([
    0.30, 0.35, 0.40, 0.50, 0.60,
    0.80, 1.00, 1.20, 1.50
])

VG = VT + overdrive
x = 1.0 / (VG - VT)  # Inverse gate overdrive (1/V)

# Synthetic deviations from the ideal straight line
noise_ohm = np.array([
    2.0, -3.0, 1.0, -2.0, 3.0,
    -1.0, 2.0, -2.0, 0.0
])

# Total resistance (synthetic data)
Rtot = Rext_true + k_true * x + noise_ohm

# 3. Linear regression: Rtot = intercept + slope*x
slope, intercept = np.polyfit(x, Rtot, 1)

Rext = intercept
RLDD = (Rext - R_NiSi - 2.0 * R_SD) / 2.0

# 4. Example operating point
example_overdrive = 0.50
example_Rtot = Rext_true + k_true / example_overdrive
example_ID = VDS / example_Rtot

# 5. Print results
print("ILLUSTRATIVE EXAMPLE: synthetic data, not measurements")
print(f"VDS = {VDS * 1000:.1f} mV")
print(f"Gate overdrive = {example_overdrive:.2f} V")
print(f"Example total resistance = {example_Rtot:.2f} ohm")
print(f"Example drain current = {example_ID * 1e6:.2f} microamp")
print()
print(f"Fitted slope k = {slope:.3f} ohm*V")
print(f"Fitted intercept Rext = {Rext:.3f} ohm")
print(f"Assumed R_NiSi = {R_NiSi:.2f} ohm")
print(f"Assumed R_SD per side = {R_SD:.2f} ohm")
print(f"Extracted RLDD = {RLDD:.3f} ohm")

# 6. Plot the data and fitted line
x_fit = np.linspace(0.0, max(x) * 1.08, 250)
R_fit = intercept + slope * x_fit

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(x, Rtot, label="Illustrative synthetic data")
ax.plot(x_fit, R_fit, label="Linear fit", linewidth=2)

ax.axhline(
    intercept,
    linestyle="--",
    linewidth=1.2,
    label=f"Y-intercept = {intercept:.1f} ohm"
)

ax.scatter([0], [intercept], zorder=4)

ax.annotate(
    f"Rext = {intercept:.1f} ohm",
    xy=(0, intercept),
    xytext=(0.65, intercept + 70),
    arrowprops={"arrowstyle": "->"}
)

ax.set_xlabel(r"$1/(V_G - V_T)$ (V$^{-1}$)")
ax.set_ylabel(r"$R_{\mathrm{tot}}=V_{DS}/I_D$ ($\Omega$)")
ax.set_title("Illustrative LDD Resistance Extraction")
ax.grid(True, alpha=0.3)
ax.legend()

# Robust automatic figure layout
fig.set_constrained_layout(True)

fig.savefig(
    "ldd_resistance_extraction_example.png",
    dpi=200,
    bbox_inches="tight"
)

print("\nPlot saved as ldd_resistance_extraction_example.png")
plt.show()
