import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Set style for clean, professional look
plt.style.use('default')
sns.set_palette("viridis")

# Generate 10,000 data points for realistic MA plot
np.random.seed(42)  # For reproducible results

# Generate realistic MA plot data with proper variance structure
n_points = 10000

# Generate average expression values (A values) - log2 of average counts
# Most genes have low expression, fewer have high expression (log-normal distribution)
# Adjust to avoid the line of points at high expression
avg_expr = np.random.lognormal(mean=1.2, sigma=1.0, size=n_points)
avg_expr = np.clip(avg_expr, 0.1, 12)  # Reduced upper limit to avoid line effect

# Generate log2 fold changes (M values) with proper variance structure
# Variance should be highest at low expression and decrease with higher expression
log2fc = np.zeros(n_points)

for i in range(n_points):
    # Calculate variance based on expression level (higher variance at low expression)
    # This mimics the real behavior in RNA-seq data - more pronounced funnel
    expr_level = avg_expr[i]
    if expr_level < 0.3:
        variance = 4.0  # Very high variance for very low expression
    elif expr_level < 1:
        variance = 3.5  # High variance for low expression
    elif expr_level < 3:
        variance = 2.5  # Medium-high variance for moderate expression
    elif expr_level < 6:
        variance = 1.5  # Medium variance for higher expression
    else:
        variance = 0.8  # Lower variance for high expression
    
    # Generate fold change with expression-dependent variance
    # Most genes should cluster tightly around zero with more pronounced funnel
    if np.random.random() < 0.90:  # 90% of genes have small changes
        log2fc[i] = np.random.normal(0, variance * 0.2)  # Even tighter clustering
    else:
        log2fc[i] = np.random.normal(0, variance * 0.8)  # Some spread but controlled

# Add some differentially expressed genes
# Up-regulated genes (positive log2 fold change)
n_up = 350
up_indices = np.random.choice(n_points, n_up, replace=False)
log2fc[up_indices] = np.random.normal(3.0, 0.5, n_up)

# Down-regulated genes (negative log2 fold change)
n_down = 350
down_indices = np.random.choice(np.setdiff1d(np.arange(n_points), up_indices), n_down, replace=False)
log2fc[down_indices] = np.random.normal(-3.0, 0.5, n_down)

# Create the figure
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

# Calculate density for coloring
xy = np.vstack([avg_expr, log2fc])
z = gaussian_kde(xy)(xy)

# Create scatter plot with density-based coloring
scatter = ax.scatter(avg_expr, log2fc, c=z, cmap='viridis', 
                    s=12, alpha=1.0, edgecolors='none')  # Larger markers, no transparency

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
cbar.set_label('Density', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# Customize the plot
ax.set_xlabel('Average Expression (A)', fontsize=14, fontweight='bold')
ax.set_ylabel('Log2 Fold Change (M)', fontsize=14, fontweight='bold')
ax.set_title('Differential Expression Analysis (MA Plot)', fontsize=16, fontweight='bold', pad=20)

# Add reference lines
ax.axhline(y=0, color='#666666', linestyle='-', linewidth=1.2, alpha=0.8)
ax.axhline(y=1, color='#666666', linestyle='--', linewidth=0.8, alpha=0.6)
ax.axhline(y=-1, color='#666666', linestyle='--', linewidth=0.8, alpha=0.6)

# Set axis limits and use log scale for x-axis
ax.set_xlim(0.1, 12)
ax.set_ylim(-5, 5)
ax.set_xscale('log')  # Log scale for expression values

# Remove spines and customize grid
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# Customize grid
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)

# Format x-axis ticks for log scale
ax.xaxis.set_major_formatter(plt.ScalarFormatter())
ax.xaxis.set_minor_formatter(plt.ScalarFormatter())

# Adjust layout and save
plt.tight_layout()
plt.savefig('ma_plot.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("MA plot generated successfully as ma_plot.png")
print(f"Generated {n_points} data points")
print(f"Added {n_up} up-regulated and {n_down} down-regulated genes")
print("Variance structure: More pronounced funnel shape")
print("Distribution: Classic MA plot with tight clustering and funnel spread") 