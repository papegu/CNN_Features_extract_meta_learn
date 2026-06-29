import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 10

# Données SHAP (top 20 features)
data = {
    'Feature': [
        'ResNet50_22', 'ResNet50_14', 'ResNet50_4', 'ResNet50_7',
        'EfficientNetB0_11', 'EfficientNetB0_23', 'MobileNetV2_8', 'ResNet50_18',
        'EfficientNetB0_15', 'MobileNetV2_19', 'ResNet50_31', 'EfficientNetB0_7',
        'MobileNetV2_12', 'ResNet50_9', 'EfficientNetB0_19', 'ResNet50_26',
        'MobileNetV2_4', 'ResNet50_11', 'EfficientNetB0_3', 'ResNet50_16'
    ],
    'SHAP_Value': [
        0.100, 0.085, 0.082, 0.078,
        0.065, 0.062, 0.058, 0.054,
        0.051, 0.048, 0.045, 0.042,
        0.039, 0.036, 0.033, 0.031,
        0.028, 0.025, 0.022, 0.020
    ],
    'Backbone': [
        'ResNet50', 'ResNet50', 'ResNet50', 'ResNet50',
        'EfficientNetB0', 'EfficientNetB0', 'MobileNetV2', 'ResNet50',
        'EfficientNetB0', 'MobileNetV2', 'ResNet50', 'EfficientNetB0',
        'MobileNetV2', 'ResNet50', 'EfficientNetB0', 'ResNet50',
        'MobileNetV2', 'ResNet50', 'EfficientNetB0', 'ResNet50'
    ]
}

df = pd.DataFrame(data)

# Couleurs par backbone
colors = {
    'ResNet50': '#1E88E5',      # Bleu
    'EfficientNetB0': '#FF8C00', # Orange
    'MobileNetV2': '#2E7D32'     # Vert
}

# Trier par valeur SHAP
df = df.sort_values('SHAP_Value', ascending=True)

# Créer le graphique
fig, ax = plt.subplots(figsize=(10, 8))

# Barres horizontales
bars = ax.barh(range(len(df)), df['SHAP_Value'], 
               color=[colors[b] for b in df['Backbone']])

# Personnalisation
ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Most Important Features (SHAP Analysis)', 
             fontsize=14, fontweight='bold', pad=20)

# Y-axis labels
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df['Feature'], fontsize=9)

# Ajouter les valeurs sur les barres
for i, (bar, val) in enumerate(zip(bars, df['SHAP_Value'])):
    width = bar.get_width()
    ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
            f'{val:.3f}', ha='left', va='center', fontsize=8)

# Légende
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors['ResNet50'], label='ResNet50'),
                   plt.Rectangle((0,0),1,1, facecolor=colors['EfficientNetB0'], label='EfficientNetB0'),
                   plt.Rectangle((0,0),1,1, facecolor=colors['MobileNetV2'], label='MobileNetV2')]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=True)

# Grille
ax.grid(True, alpha=0.3, axis='x')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('shap_importance_improved.png', dpi=300, bbox_inches='tight')
plt.show()