import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. BPE Tokenization Graphic
# ==========================================
# Set up the figure
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
ax.axis('off')

# ----------------- Boxes -----------------
# 1. Top Left: Incorrect NLLB Output (Red outline)
ax.add_patch(plt.Rectangle((1, 3), 2, 0.8, facecolor='#f2dede', edgecolor='#d9534f', lw=2, zorder=2))
ax.text(2, 3.4, "kumelwe", ha='center', va='center', fontsize=16, fontweight='bold', zorder=3)
ax.text(2, 4.0, "NLLB Output", ha='center', va='center', fontsize=11, color='#d9534f', fontweight='bold')

# 2. Top Right: Correct Morphology (Green outline)
ax.add_patch(plt.Rectangle((6, 3), 2, 0.8, facecolor='#dff0d8', edgecolor='#5cb85c', lw=2, zorder=2))
ax.text(7, 3.4, "kumele", ha='center', va='center', fontsize=16, fontweight='bold', zorder=3)
ax.text(7, 4.0, "Correct isiZulu", ha='center', va='center', fontsize=11, color='#5cb85c', fontweight='bold')

# 3. Bottom: BPE Subwords (Yellow outline)
ax.add_patch(plt.Rectangle((0.5, 1), 1.5, 0.8, facecolor='#fcf8e3', edgecolor='#f0ad4e', lw=2, zorder=2))
ax.text(1.25, 1.4, "kume", ha='center', va='center', fontsize=16, fontweight='bold', zorder=3)

ax.add_patch(plt.Rectangle((3, 1), 1.5, 0.8, facecolor='#fcf8e3', edgecolor='#f0ad4e', lw=2, zorder=2))
ax.text(3.75, 1.4, "lwe", ha='center', va='center', fontsize=16, fontweight='bold', zorder=3)

ax.text(2.5, 1.4, "+", ha='center', va='center', fontsize=20, fontweight='bold', zorder=3)

# ----------------- Arrows -----------------
# Arrow from NLLB Output down to BPE split
ax.annotate("", xy=(2, 2.0), xytext=(2, 2.8), arrowprops=dict(arrowstyle="-|>", lw=2, color='#333333'))

# Arrow from NLLB Output to Correct Output (Dashed)
ax.annotate("", xy=(6, 3.4), xytext=(3.2, 3.4), arrowprops=dict(arrowstyle="-|>", lw=2, color='#5cb85c', linestyle='dashed'))

# ----------------- Text Annotations -----------------
# Text under BPE split
ax.text(2, 0.5, "Statistical subwords (ignores morphology)", ha='center', va='center', fontsize=11, style='italic', color='#555555')

# Text above dashed arrow
ax.text(4.6, 3.6, "Fails to map to", ha='center', va='center', fontsize=10, color='#5cb85c')
ax.text(4.6, 3.2, "correct grammar", ha='center', va='center', fontsize=10, color='#5cb85c')

# Title
ax.text(4.5, 4.8, "Byte-Pair Encoding (BPE) Failure Mode", ha='center', va='center', fontsize=16, fontweight='bold', color='#1e293b')

# Set limits and save
plt.xlim(0, 9)
plt.ylim(0, 5.5)
plt.tight_layout()
plt.savefig("bpe_graphic.png", dpi=300, bbox_inches='tight')
plt.close()

print("Perfect BPE graphic generated successfully!")

# ==========================================
# 2. Model Memory Footprint Chart
# ==========================================
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

models = ['FP32\n(Baseline)', 'INT8\n(PyTorch/CT2)', 'INT4\n(Unsupported)']
memory = [2.4, 0.6, 0.3] # in GB
colors = ['#d9534f', '#5cb85c', '#f0ad4e']

bars = ax.bar(models, memory, color=colors, width=0.5)

# Add Pepper's RAM limit line
ax.axhline(4.0, color='red', linestyle='--', linewidth=1.5, label='Pepper RAM Limit (4GB)')
# Add a visual marker for where the model sits in the 4GB limit
ax.axhline(2.4, color='#d9534f', linestyle=':', linewidth=1)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval} GB", ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_ylabel('Memory Footprint (GB)', fontsize=11, fontweight='bold')
ax.set_title('NLLB-200 Memory Footprint vs. Pepper RAM', fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, 4.5)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("memory_chart.png", dpi=300, bbox_inches='tight')
plt.close()

print("Images generated successfully!")