import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, expon
from scipy.optimize import curve_fit

# Set style for clean, professional look
plt.style.use('default')
sns.set_palette("viridis")

# Generate synthetic data for mixed model
np.random.seed(42)

# Parameters for the mixed model
n_samples = 5000

# Generate data from multiple components
# Component 1: Gaussian (main peak)
gauss1_data = np.random.normal(loc=2.5, scale=0.8, size=int(n_samples * 0.4))

# Component 2: Gaussian (secondary peak)
gauss2_data = np.random.normal(loc=5.0, scale=1.2, size=int(n_samples * 0.3))

# Component 3: Exponential (tail)
expon_data = np.random.exponential(scale=1.5, size=int(n_samples * 0.3))

# Combine all data
all_data = np.concatenate([gauss1_data, gauss2_data, expon_data])

# Define the mixed model function
def mixed_model(x, a1, mu1, sigma1, a2, mu2, sigma2, a3, lambda_exp):
    """
    Mixed model with 2 Gaussians + 1 exponential
    a1, a2, a3: amplitudes
    mu1, mu2: means of Gaussians
    sigma1, sigma2: standard deviations of Gaussians
    lambda_exp: rate parameter for exponential
    """
    gauss1 = a1 * np.exp(-0.5 * ((x - mu1) / sigma1)**2)
    gauss2 = a2 * np.exp(-0.5 * ((x - mu2) / sigma2)**2)
    expon_comp = a3 * lambda_exp * np.exp(-lambda_exp * x)
    return gauss1 + gauss2 + expon_comp

# Create histogram data for fitting
hist, bin_edges = np.histogram(all_data, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Initial parameter guesses
p0 = [0.3, 2.5, 0.8, 0.2, 5.0, 1.2, 0.1, 0.5]

# Fit the model
try:
    popt, pcov = curve_fit(mixed_model, bin_centers, hist, p0=p0, 
                          bounds=([0, 0, 0.1, 0, 0, 0.1, 0, 0.1], 
                                 [1, 10, 3, 1, 10, 3, 1, 2]))
except:
    # If fitting fails, use initial parameters
    popt = p0

# Create the figure with standardized size
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

# Plot histogram with fitted model
ax.hist(all_data, bins=50, density=True, alpha=0.7, color='#3b82f6', 
        edgecolor='white', linewidth=0.5, label='Data')

# Plot fitted model
x_fit = np.linspace(0, 12, 200)
y_fit = mixed_model(x_fit, *popt)
ax.plot(x_fit, y_fit, 'r-', linewidth=2.5, label='Fitted Model')

# Plot individual components
gauss1_fit = popt[0] * np.exp(-0.5 * ((x_fit - popt[1]) / popt[2])**2)
gauss2_fit = popt[3] * np.exp(-0.5 * ((x_fit - popt[4]) / popt[5])**2)
expon_fit = popt[6] * popt[7] * np.exp(-popt[7] * x_fit)

ax.plot(x_fit, gauss1_fit, '--', color='#10b981', linewidth=1.5, alpha=0.8, label='Gaussian 1')
ax.plot(x_fit, gauss2_fit, '--', color='#f59e0b', linewidth=1.5, alpha=0.8, label='Gaussian 2')
ax.plot(x_fit, expon_fit, '--', color='#ef4444', linewidth=1.5, alpha=0.8, label='Exponential')

ax.set_xlabel('Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12, fontweight='bold')
ax.set_title('Mixed Model: Data + Fitted Components', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 12)

# Add model parameters as text
param_text = f'Model Parameters:\n'
param_text += f'Gaussian 1: μ={popt[1]:.2f}, σ={popt[2]:.2f}\n'
param_text += f'Gaussian 2: μ={popt[4]:.2f}, σ={popt[5]:.2f}\n'
param_text += f'Exponential: λ={popt[7]:.2f}'

fig.text(0.02, 0.02, param_text, fontsize=10, fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('mixed_model.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("Mixed model visualization generated successfully as mixed_model.png")
print(f"Generated {n_samples} data points from mixed distribution")
print("Components: 2 Gaussians + 1 Exponential")
print("Standardized size for consistent card layout") 