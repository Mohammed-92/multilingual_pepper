import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Widened the figure to accommodate the text
fig, ax = plt.subplots(figsize=(12, 3), dpi=300)
ax.axis('off')

# Adjusted x-positions to space the boxes evenly
boxes = [
    ("English Input\n(Sentence)", 1.5, "#d9edf7"),
    ("NLLB-200\n(Translation Model)", 4.5, "#dff0d8"),
    ("Target Output\n(isiZulu / Afrikaans /\nN. Sotho)", 7.5, "#d9edf7"), # Wrapped the text here
    ("Automatic Evaluation\n(BLEU & Latency)", 10.5, "#fcf8e3")
]

box_width = 2.4
box_height = 1.0
y_pos = 0.4

for i, (text, x, color) in enumerate(boxes):
    # Draw the box
    ax.add_patch(plt.Rectangle((x - box_width/2, y_pos), box_width, box_height, 
                               facecolor=color, edgecolor='black', lw=1.5, zorder=2))
    
    # Add the text
    ax.text(x, y_pos + box_height/2, text, ha='center', va='center', 
            fontsize=11, fontweight='bold', zorder=3)
    
    # Draw the arrow to the next box (preventing overlap)
    if i < len(boxes) - 1:
        next_x = boxes[i+1][1]
        start_arrow = x + box_width/2
        end_arrow = next_x - box_width/2
        ax.annotate("", xy=(end_arrow, y_pos + box_height/2), xytext=(start_arrow, y_pos + box_height/2), 
                    arrowprops=dict(arrowstyle="-|>", lw=2, color='#333333'))

# Add the note at the bottom
ax.text(6.0, 0.1, "Note: ASR and TTS are conceptual components not evaluated in this study.", 
        ha='center', va='center', fontsize=10, style='italic', color='#555555')

plt.xlim(0, 12)
plt.ylim(0, 1.5)
plt.tight_layout()
plt.savefig("pipeline.png", dpi=300, bbox_inches='tight')
plt.close()

print("Fixed pipeline.png generated successfully!")