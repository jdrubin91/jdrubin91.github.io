import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for the normal distribution
mu = 0  # mean
sigma = 1  # standard deviation

# Generate the x values for the normal distribution
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
y = norm.pdf(x, mu, sigma)

# Find a sample that would result in a p-value of less than 0.01 (two-tailed)
critical_z = norm.ppf(0.01 / 2)

# Generate the corresponding y value for the sample
sample = critical_z
p_value = 2 * norm.cdf(sample)  # two-tailed p-value

# Confidence interval boundaries for 95% confidence (two-tailed, 2.5% in each tail)
ci_lower = norm.ppf(0.025, mu, sigma)
ci_upper = norm.ppf(0.975, mu, sigma)

# Plotting the normal distribution
F = plt.figure(figsize=(8, 6))
ax = F.add_subplot(111)
ax.plot(x, y, label='Normal Distribution', color='blue')

# Highlight the critical region that contributes to the p-value
x_fill = np.linspace(-4, sample, 100)
ax.fill_between(x_fill, norm.pdf(x_fill), color='orange', alpha=0.5, label=f'P-value area: {p_value:.2f}')

# Plot the sample as a red line
ax.axvline(sample, color='red', linestyle='--', label=f'Sample (z={sample:.2f})')

# # Add the 95% confidence interval as a shaded region
# ax.fill_betweenx(y, ci_lower, ci_upper, color='green', alpha=0.5, label='95% Confidence Interval')

# Add text with the exact p-value on the plot
ax.text(sample-0.1, 0.05, f'p-value = {p_value:.2f}', color='red', fontsize=12, ha='right')

# Labels and legend
# ax.title('Normal Distribution with Sample Leading to p-value < 0.01 and 95% Confidence Interval')
ax.set_xlabel('Value')
ax.set_ylabel('Probability Density')
# ax.legend()
ax.spines[['right', 'top']].set_visible(False)

# Show the plot
plt.grid(False)
plt.tight_layout()

plt.savefig('pval.png')
