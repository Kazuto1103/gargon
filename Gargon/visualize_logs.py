import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if hasattr(plt.style, 'seaborn-v0_8-whitegrid') else 'default')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Data from experiment reports
experiment_data = {
    'RUN 1': {
        'params': {'GLCM d': 1, 'LBP R': 1, 'Epochs': 8},
        'train_acc': 0.9784,
        'test_acc': 0.9803,
        'final_loss': 0.058182,
        'loss_per_epoch': [0.361187, 0.159027, 0.117728, 0.096829, 0.096581, 0.715250, 0.731272, 0.775400]
    },
    'RUN 2': {
        'params': {'GLCM d': 1, 'LBP R': 1, 'Epochs': 15},
        'train_acc': 0.9815,
        'test_acc': 0.9826,
        'final_loss': 0.038988,
        'loss_per_epoch': [0.361187, 0.159027, 0.117728, 0.096829, 0.096581, 0.715250, 0.731272, 0.775400,
                          0.757977, 0.780795, 0.777627, 0.785531, 0.788514, 0.797677, 0.766865]
    },
    'RUN 3': {
        'params': {'GLCM d': 3, 'LBP R': 2, 'Epochs': 15},
        'train_acc': 0.9969,
        'test_acc': 0.9815,
        'final_loss': 0.016955,
        'loss_per_epoch': [0.299911, 0.099823, 0.068804, 0.051361, 0.043027, 0.036844, 0.035159, 0.029619,
                          0.026643, 0.023177, 0.019342, 0.019003, 0.017069, 0.018559, 0.016955]
    }
}

# Data distribution (from experiment report)
class_distribution = {'Kelas 0 (Nominal)': 1920, 'Kelas 1 (Anomalous)': 2400}
train_test_split = {'Training (60%)': 2592, 'Testing (40%)': 1728}

logs_dir = Path('d:/Project/Gargon/logs')
output_dir = Path('d:/Project/Gargon/visualizations')
output_dir.mkdir(exist_ok=True)

# Visualization 1: Train and Test Split (6:4)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart for train/test split
colors = ['#3498db', '#e74c3c']
ax1.pie(train_test_split.values(), labels=train_test_split.keys(), autopct='%1.1f%%',
        colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
ax1.set_title('Data Split: Training vs Testing (60:40)', fontsize=14, fontweight='bold', pad=20)

# Bar chart for class distribution
classes = list(class_distribution.keys())
counts = list(class_distribution.values())
bars = ax2.bar(classes, counts, color=['#2ecc71', '#f39c12'])
ax2.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold', pad=20)
ax2.set_ylabel('Number of Samples', fontsize=12)
ax2.set_xlabel('Class', fontsize=12)
ax2.bar_label(bars, padding=3, fontsize=11, fontweight='bold')
ax2.set_ylim(0, max(counts) * 1.1)

plt.tight_layout()
plt.savefig(output_dir / '1_train_test_split.png', dpi=300, bbox_inches='tight')
print("✓ Visualization 1 saved: Train/Test Split")
plt.close()

# Visualization 2: Comparison of Training and Testing Sessions
fig, ax = plt.subplots(figsize=(14, 8))

runs = list(experiment_data.keys())
train_accs = [experiment_data[run]['train_acc'] * 100 for run in runs]
test_accs = [experiment_data[run]['test_acc'] * 100 for run in runs]

x = np.arange(len(runs))
width = 0.35

bars1 = ax.bar(x - width/2, train_accs, width, label='Training Accuracy', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, test_accs, width, label='Testing Accuracy', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Experiment Runs', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Training vs Testing Accuracy Comparison Across Runs', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([f'{run}\n(GLCM d={experiment_data[run]["params"]["GLCM d"]}, LBP R={experiment_data[run]["params"]["LBP R"]})' 
                    for run in runs], fontsize=10)
ax.legend(fontsize=11, loc='lower right')
ax.set_ylim(95, 100)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '2_train_test_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualization 2 saved: Train/Test Comparison")
plt.close()

# Visualization 3: Comparison of Each Epoch (Loss Progression)
fig, ax = plt.subplots(figsize=(14, 8))

colors_run = ['#3498db', '#e74c3c', '#2ecc71']
markers = ['o', 's', '^']

for idx, (run_name, data) in enumerate(experiment_data.items()):
    epochs = range(1, len(data['loss_per_epoch']) + 1)
    losses = data['loss_per_epoch']
    ax.plot(epochs, losses, marker=markers[idx], color=colors_run[idx], 
            linewidth=2.5, markersize=8, label=f'{run_name} (d={data["params"]["GLCM d"]}, R={data["params"]["LBP R"]})')

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('Loss Progression Across Epochs for Each Run', fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(output_dir / '3_epoch_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualization 3 saved: Epoch Comparison")
plt.close()

# Visualization 4: Telemetry Data - Combined Line Chart
telemetry_files = sorted(logs_dir.glob('telemetry_*.csv'))

if telemetry_files:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Telemetry Data Analysis - All Sessions Combined', fontsize=16, fontweight='bold')
    
    all_data = []
    session_names = []
    
    for telemetry_file in telemetry_files:
        try:
            df = pd.read_csv(telemetry_file)
            if len(df) > 2:  # Skip empty or minimal files
                df['session'] = telemetry_file.stem
                all_data.append(df)
                session_names.append(telemetry_file.stem)
        except Exception as e:
            print(f"  Skipping {telemetry_file.name}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])
        
        # Plot 1: Processing Time over Time
        ax1 = axes[0, 0]
        for session in session_names[:8]:  # Limit to first 8 sessions for clarity
            session_data = combined_df[combined_df['session'] == session]
            if len(session_data) > 0:
                ax1.plot(range(len(session_data)), session_data['Processing_Time_ms'], 
                        alpha=0.6, linewidth=1, label=session)
        ax1.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Processing Time (ms)', fontsize=11, fontweight='bold')
        ax1.set_title('Processing Time per Frame', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8, loc='upper right', ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence Score Distribution
        ax2 = axes[0, 1]
        confidence_data = combined_df['Confidence_Score'].dropna()
        ax2.hist(confidence_data, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Confidence Score (%)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Confidence Score Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Predicted Class Distribution
        ax3 = axes[1, 0]
        class_counts = combined_df['Predicted_Class'].value_counts().sort_index()
        colors_class = ['#2ecc71', '#e74c3c']
        bars = ax3.bar(['Nominal (0)', 'Anomalous (1)'], 
                       [class_counts.get(0, 0), class_counts.get(1, 0)],
                       color=colors_class, alpha=0.8)
        ax3.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax3.set_title('Predicted Class Distribution', fontsize=12, fontweight='bold')
        ax3.bar_label(bars, padding=3, fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Processing Time Statistics per Session
        ax4 = axes[1, 1]
        session_stats = []
        session_labels = []
        for session in session_names[:10]:
            session_data = combined_df[combined_df['session'] == session]
            if len(session_data) > 0:
                session_stats.append(session_data['Processing_Time_ms'].describe())
                session_labels.append(session[-12:])  # Shorten name
        
        if session_stats:
            stats_df = pd.DataFrame(session_stats, index=session_labels)
            x_pos = np.arange(len(session_labels))
            ax4.bar(x_pos, stats_df['mean'], yerr=stats_df['std'], 
                   capsize=5, color='#9b59b6', alpha=0.8)
            ax4.set_xlabel('Session', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Mean Processing Time (ms)', fontsize=11, fontweight='bold')
            ax4.set_title('Processing Time per Session (Mean ± Std)', fontsize=12, fontweight='bold')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(session_labels, rotation=45, ha='right', fontsize=8)
            ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_telemetry_combined.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization 4 saved: Telemetry Combined")
    plt.close()
else:
    print("⚠ No telemetry files found")

print(f"\n✅ All visualizations saved to: {output_dir}")
print("\nGenerated files:")
print("  1. train_test_split.png - Data split visualization")
print("  2. train_test_comparison.png - Training vs Testing accuracy")
print("  3. epoch_comparison.png - Loss progression across epochs")
print("  4. telemetry_combined.png - Combined telemetry analysis")
