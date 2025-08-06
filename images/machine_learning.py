import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for clean, professional look
plt.style.use('default')

# Create the figure
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

# Define colors
colors = {
    'primary': '#3b82f6',
    'secondary': '#10b981', 
    'accent': '#f59e0b',
    'warning': '#ef4444',
    'light': '#f8fafc',
    'dark': '#1f2937'
}

# ML Model Performance data
np.random.seed(42)

models = ['Random\nForest', 'Neural\nNetwork', 'SVM', 'XGBoost', 'Ensemble']
accuracy = [0.85, 0.92, 0.78, 0.89, 0.94]
precision = [0.83, 0.90, 0.75, 0.87, 0.92]
recall = [0.87, 0.91, 0.80, 0.88, 0.93]

x = np.arange(len(models))
width = 0.25

# Create grouped bar chart
bars1 = ax.bar(x - width, accuracy, width, label='Accuracy', color=colors['primary'], alpha=0.8)
bars2 = ax.bar(x, precision, width, label='Precision', color=colors['secondary'], alpha=0.8)
bars3 = ax.bar(x + width, recall, width, label='Recall', color=colors['accent'], alpha=0.8)

# Customize the plot
ax.set_xlabel('Machine Learning Models', fontsize=12, fontweight='bold')
ax.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add ML workflow indicators
workflow_text = "ML Workflow:\n• Data Preprocessing\n• Feature Engineering\n• Model Training\n• Validation\n• Deployment"

fig.text(0.02, 0.02, workflow_text, fontsize=10, fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['light'], alpha=0.8))

plt.tight_layout()
plt.savefig('machine_learning.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("Machine learning visualization generated successfully as machine_learning.png")
print("Single figure showing model performance comparison")
print("Standardized size for consistent card layout") 