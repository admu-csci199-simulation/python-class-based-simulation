import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Data entry - manually entered from the table
data = {
    'Initial_Red': [100, 200, 50, 50, 100, 200, 50, 50, 100, 200, 50, 50, 100, 200, 50, 50, 100, 200, 50, 50, 100, 200, 50, 50, 100, 200, 50, 50, 100, 200, 50, 50, 100, 200, 50, 50],
    'Initial_Centrist': [100, 50, 200, 50, 100, 50, 200, 50, 100, 50, 200, 50, 100, 50, 200, 50, 100, 50, 200, 50, 100, 50, 200, 50, 100, 50, 200, 50, 100, 50, 200, 50, 100, 50, 200, 50],
    'Initial_Blue': [100, 50, 50, 200, 100, 50, 50, 200, 100, 50, 50, 200, 100, 50, 50, 200, 100, 50, 50, 200, 100, 50, 50, 200, 100, 50, 50, 200, 100, 50, 50, 200, 100, 50, 50, 200],
    'Post_Value': [-4, -4, -4, -4, -3, -3, -3, -3, -2, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
    'Post_Label': ['Far Red', 'Far Red', 'Far Red', 'Far Red', 'Very Red', 'Very Red', 'Very Red', 'Very Red', 
                   'Moderately Red', 'Moderately Red', 'Moderately Red', 'Moderately Red',
                   'Red-leaning Centrist', 'Red-leaning Centrist', 'Red-leaning Centrist', 'Red-leaning Centrist',
                   'Centrist', 'Centrist', 'Centrist', 'Centrist',
                   'Blue-leaning Centrist', 'Blue-leaning Centrist', 'Blue-leaning Centrist', 'Blue-leaning Centrist',
                   'Moderately Blue', 'Moderately Blue', 'Moderately Blue', 'Moderately Blue',
                   'Very Blue', 'Very Blue', 'Very Blue', 'Very Blue',
                   'Far Blue', 'Far Blue', 'Far Blue', 'Far Blue'],
    'Change_Red': [55, 29, 113, 23, 54, 128, 29, 61, 124, 32, -58, -127, -26, -64, -118, -29, -56, -115, -31, -57, -125, -28, -28, -25, -56, -11, -15, -22, -44, -17, -14, -28, -44, -17, -14, -28],
    'Change_Centrist': [-29, -22, -98, 26, -25, -118, 4, -10, -104, 84, 108, 159, 63, 146, 151, 60, 120, 150, 65, 161, 95, -105, -8, 26, -101, -14, -18, 20, -124, -18, -29, -76, -26, 30, 14, -18],
    'Change_Blue': [-26, -7, -15, -49, -29, -10, -33, -51, -20, -116, -50, -32, -27, -112, -33, -31, -64, -35, -35, -134, 30, 133, 36, -51, 36, 112, 29, 31, 141, 32, 45, 58, 56, -50, -22, -8]
}

df = pd.DataFrame(data)

# Create distribution labels
df['Distribution_Type'] = df.apply(lambda row: 
    'Red Majority' if row['Initial_Red'] > row['Initial_Blue'] and row['Initial_Red'] > row['Initial_Centrist']
    else 'Blue Majority' if row['Initial_Blue'] > row['Initial_Red'] and row['Initial_Blue'] > row['Initial_Centrist']
    else 'Centrist Majority' if row['Initial_Centrist'] > row['Initial_Red'] and row['Initial_Centrist'] > row['Initial_Blue']
    else 'Balanced', axis=1)

# Create figure with subplots
fig = plt.figure(figsize=(18, 14))

# 1. Heatmap of Red camp changes
ax1 = plt.subplot(3, 3, 1)
pivot_red = df.pivot_table(values='Change_Red', index='Post_Value', columns='Distribution_Type', aggfunc='mean')
sns.heatmap(pivot_red, annot=True, fmt='.0f', cmap='RdYlGn', center=0, cbar_kws={'label': 'Change in Red Count'})
ax1.set_title('Red Camp Changes by Post Value & Distribution', fontweight='bold', fontsize=12)
ax1.set_xlabel('Distribution Type')
ax1.set_ylabel('Post Value')

# 2. Heatmap of Blue camp changes
ax2 = plt.subplot(3, 3, 2)
pivot_blue = df.pivot_table(values='Change_Blue', index='Post_Value', columns='Distribution_Type', aggfunc='mean')
sns.heatmap(pivot_blue, annot=True, fmt='.0f', cmap='RdYlGn', center=0, cbar_kws={'label': 'Change in Blue Count'})
ax2.set_title('Blue Camp Changes by Post Value & Distribution', fontweight='bold', fontsize=12)
ax2.set_xlabel('Distribution Type')
ax2.set_ylabel('Post Value')

# 3. Heatmap of Centrist changes
ax3 = plt.subplot(3, 3, 3)
pivot_centrist = df.pivot_table(values='Change_Centrist', index='Post_Value', columns='Distribution_Type', aggfunc='mean')
sns.heatmap(pivot_centrist, annot=True, fmt='.0f', cmap='RdYlGn', center=0, cbar_kws={'label': 'Change in Centrist Count'})
ax3.set_title('Centrist Changes by Post Value & Distribution', fontweight='bold', fontsize=12)
ax3.set_xlabel('Distribution Type')
ax3.set_ylabel('Post Value')

# 4. Line plot - Changes by post value for Red Majority scenarios
ax4 = plt.subplot(3, 3, 4)
red_maj = df[df['Distribution_Type'] == 'Red Majority'].groupby('Post_Value')[['Change_Red', 'Change_Blue', 'Change_Centrist']].mean()
ax4.plot(red_maj.index, red_maj['Change_Red'], marker='o', linewidth=2, label='Red', color='red')
ax4.plot(red_maj.index, red_maj['Change_Blue'], marker='s', linewidth=2, label='Blue', color='blue')
ax4.plot(red_maj.index, red_maj['Change_Centrist'], marker='^', linewidth=2, label='Centrist', color='gray')
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax4.set_xlabel('Post Value')
ax4.set_ylabel('Average Change in Count')
ax4.set_title('Red Majority Scenarios', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Line plot - Changes by post value for Blue Majority scenarios
ax5 = plt.subplot(3, 3, 5)
blue_maj = df[df['Distribution_Type'] == 'Blue Majority'].groupby('Post_Value')[['Change_Red', 'Change_Blue', 'Change_Centrist']].mean()
ax5.plot(blue_maj.index, blue_maj['Change_Red'], marker='o', linewidth=2, label='Red', color='red')
ax5.plot(blue_maj.index, blue_maj['Change_Blue'], marker='s', linewidth=2, label='Blue', color='blue')
ax5.plot(blue_maj.index, blue_maj['Change_Centrist'], marker='^', linewidth=2, label='Centrist', color='gray')
ax5.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax5.set_xlabel('Post Value')
ax5.set_ylabel('Average Change in Count')
ax5.set_title('Blue Majority Scenarios', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Line plot - Changes by post value for Centrist Majority scenarios
ax6 = plt.subplot(3, 3, 6)
cent_maj = df[df['Distribution_Type'] == 'Centrist Majority'].groupby('Post_Value')[['Change_Red', 'Change_Blue', 'Change_Centrist']].mean()
ax6.plot(cent_maj.index, cent_maj['Change_Red'], marker='o', linewidth=2, label='Red', color='red')
ax6.plot(cent_maj.index, cent_maj['Change_Blue'], marker='s', linewidth=2, label='Blue', color='blue')
ax6.plot(cent_maj.index, cent_maj['Change_Centrist'], marker='^', linewidth=2, label='Centrist', color='gray')
ax6.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax6.set_xlabel('Post Value')
ax6.set_ylabel('Average Change in Count')
ax6.set_title('Centrist Majority Scenarios', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Bar plot - Net effect by post value (Red perspective)
ax7 = plt.subplot(3, 3, 7)
red_maj_grouped = df[df['Distribution_Type'] == 'Red Majority'].groupby('Post_Value')[['Change_Red', 'Change_Blue', 'Change_Centrist']].mean()
x = np.arange(len(red_maj_grouped))
width = 0.25
ax7.bar(x - width, red_maj_grouped['Change_Red'], width, label='Red', color='red', alpha=0.7)
ax7.bar(x, red_maj_grouped['Change_Centrist'], width, label='Centrist', color='gray', alpha=0.7)
ax7.bar(x + width, red_maj_grouped['Change_Blue'], width, label='Blue', color='blue', alpha=0.7)
ax7.set_xlabel('Post Value')
ax7.set_ylabel('Average Change')
ax7.set_title('Red Majority: Change by Camp', fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(red_maj_grouped.index)
ax7.legend()
ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax7.grid(True, alpha=0.3, axis='y')

# 8. Bar plot - Net effect by post value (Blue perspective)
ax8 = plt.subplot(3, 3, 8)
blue_maj_grouped = df[df['Distribution_Type'] == 'Blue Majority'].groupby('Post_Value')[['Change_Red', 'Change_Blue', 'Change_Centrist']].mean()
x = np.arange(len(blue_maj_grouped))
ax8.bar(x - width, blue_maj_grouped['Change_Red'], width, label='Red', color='red', alpha=0.7)
ax8.bar(x, blue_maj_grouped['Change_Centrist'], width, label='Centrist', color='gray', alpha=0.7)
ax8.bar(x + width, blue_maj_grouped['Change_Blue'], width, label='Blue', color='blue', alpha=0.7)
ax8.set_xlabel('Post Value')
ax8.set_ylabel('Average Change')
ax8.set_title('Blue Majority: Change by Camp', fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(blue_maj_grouped.index)
ax8.legend()
ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax8.grid(True, alpha=0.3, axis='y')

# 9. Summary statistics table
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

# Calculate optimal strategies
summary_text = "OPTIMAL STRATEGIES:\n\n"
summary_text += "Red Majority:\n"
red_best = red_maj_grouped.idxmax()
summary_text += f"  Best for Red: Post {red_best['Change_Red']}\n"
summary_text += f"  Worst for Red: Post {red_maj_grouped.idxmin()['Change_Red']}\n\n"

summary_text += "Blue Majority:\n"
blue_best = blue_maj_grouped.idxmax()
summary_text += f"  Best for Blue: Post {blue_best['Change_Blue']}\n"
summary_text += f"  Worst for Blue: Post {blue_maj_grouped.idxmin()['Change_Blue']}\n\n"

summary_text += "Centrist Majority:\n"
cent_best = cent_maj.idxmax()
summary_text += f"  Best for Centrist: Post {cent_best['Change_Centrist']}\n"

ax9.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top', 
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save the complete figure
plt.savefig('belief_camp_analysis_complete.png', dpi=300, bbox_inches='tight')

# Save each subplot individually
fig1 = ax1.get_figure()
extent1 = ax1.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
fig1.savefig('graph1_red_heatmap.png', dpi=300, bbox_inches=extent1.expanded(1.3, 1.3))

extent2 = ax2.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
fig1.savefig('graph2_blue_heatmap.png', dpi=300, bbox_inches=extent2.expanded(1.3, 1.3))

extent3 = ax3.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
fig1.savefig('graph3_centrist_heatmap.png', dpi=300, bbox_inches=extent3.expanded(1.3, 1.3))

extent4 = ax4.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
fig1.savefig('graph4_red_majority_lines.png', dpi=300, bbox_inches=extent4.expanded(1.3, 1.3))

extent5 = ax5.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
fig1.savefig('graph5_blue_majority_lines.png', dpi=300, bbox_inches=extent5.expanded(1.3, 1.3))

extent6 = ax6.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
fig1.savefig('graph6_centrist_majority_lines.png', dpi=300, bbox_inches=extent6.expanded(1.3, 1.3))

extent7 = ax7.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
fig1.savefig('graph7_red_majority_bars.png', dpi=300, bbox_inches=extent7.expanded(1.3, 1.3))

extent8 = ax8.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
fig1.savefig('graph8_blue_majority_bars.png', dpi=300, bbox_inches=extent8.expanded(1.3, 1.3))

extent9 = ax9.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
fig1.savefig('graph9_optimal_strategies.png', dpi=300, bbox_inches=extent9.expanded(1.3, 1.3))

plt.show()

print("All visualizations saved!")
print("Complete figure: belief_camp_analysis_complete.png")
print("Individual graphs: graph1_*.png through graph9_*.png")
print("\n=== DATA SUMMARY ===")
print(df.groupby(['Distribution_Type', 'Post_Value'])[['Change_Red', 'Change_Centrist', 'Change_Blue']].mean())