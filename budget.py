"""
generate_project_budgets.py
Génère les budgets Excel/CSV pour les 4 projets ESPOIR-Jeunes
Version avec gestion automatique des dépendances
"""

import subprocess
import sys
import os
from datetime import datetime

# Fonction pour installer les packages manquants
def install_package(package):
    print(f"📦 Installation de {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Vérifier et installer pandas si nécessaire
try:
    import pandas as pd
    print("✅ pandas déjà installé")
except ImportError:
    print("⚠️ pandas non trouvé. Installation en cours...")
    install_package('pandas')
    import pandas as pd
    print("✅ pandas installé avec succès")

# Vérifier et installer openpyxl si nécessaire
try:
    import openpyxl
    print("✅ openpyxl déjà installé")
    EXCEL_AVAILABLE = True
except ImportError:
    print("⚠️ openpyxl non trouvé. Installation en cours...")
    try:
        install_package('openpyxl')
        import openpyxl
        EXCEL_AVAILABLE = True
        print("✅ openpyxl installé avec succès")
    except:
        EXCEL_AVAILABLE = False
        print("⚠️ openpyxl n'a pas pu être installé. Utilisation du format CSV uniquement.")

# Configuration
OUTPUT_DIR = "budgets_projets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Taux de change (indicatif)
USD_TO_FCFA = 600
EURO_TO_FCFA = 655.957

class ProjectBudget:
    def __init__(self, project_name, category, total_budget, duration_months):
        self.project_name = project_name
        self.category = category
        self.total_budget = total_budget
        self.duration_months = duration_months
        self.budget_lines = []
        
    def add_line(self, category, subcategory, description, unit_cost, quantity, months=None):
        if months:
            total = unit_cost * quantity * months
        else:
            total = unit_cost * quantity
            
        self.budget_lines.append({
            'Catégorie': category,
            'Sous-catégorie': subcategory,
            'Description': description,
            'Coût unitaire (FCFA)': unit_cost,
            'Quantité': quantity,
            'Durée (mois)': months if months else '-',
            'Total (FCFA)': total
        })
        return total
    
    def display_summary(self):
        """Affiche un résumé du budget dans la console"""
        print(f"\n{'='*60}")
        print(f"📊 BUDGET: {self.project_name}")
        print(f"{'='*60}")
        print(f"Catégorie: {self.category}")
        print(f"Budget total: {self.total_budget:,.0f} FCFA")
        print(f"Durée: {self.duration_months} mois")
        print(f"{'-'*60}")
        
        # Grouper par catégorie
        categories = {}
        for line in self.budget_lines:
            cat = line['Catégorie']
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += line['Total (FCFA)']
        
        for cat, total in categories.items():
            percentage = (total / self.total_budget) * 100
            print(f"{cat}: {total:,.0f} FCFA ({percentage:.1f}%)")
        
        total_calc = sum(line['Total (FCFA)'] for line in self.budget_lines)
        print(f"{'-'*60}")
        print(f"Total calculé: {total_calc:,.0f} FCFA")
        print(f"Écart: {self.total_budget - total_calc:,.0f} FCFA")
        print(f"{'='*60}\n")
    
    def generate_excel(self):
        """Génère un fichier Excel (ou CSV si Excel non disponible)"""
        # Créer le DataFrame
        df = pd.DataFrame(self.budget_lines)
        
        # Calculer les totaux par catégorie
        totals = df.groupby('Catégorie')['Total (FCFA)'].sum().reset_index()
        totals.columns = ['Catégorie', 'Sous-total']
        
        # Ajouter une ligne de total général
        grand_total = df['Total (FCFA)'].sum()
        
        # Nom de fichier de base
        base_filename = f"{OUTPUT_DIR}/{self.project_name.replace(' ', '_').replace('-', '_')}_budget"
        
        if EXCEL_AVAILABLE:
            # Version Excel
            filename = base_filename + ".xlsx"
            try:
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    # Feuille détaillée
                    df.to_excel(writer, sheet_name='Détail', index=False)
                    
                    # Feuille résumé par catégorie
                    totals.to_excel(writer, sheet_name='Résumé par catégorie', index=False)
                    
                    # Feuille récapitulative
                    summary = pd.DataFrame({
                        'Projet': [self.project_name],
                        'Catégorie': [self.category],
                        'Budget total (FCFA)': [self.total_budget],
                        'Budget total calculé (FCFA)': [grand_total],
                        'Écart': [self.total_budget - grand_total],
                        'Durée (mois)': [self.duration_months],
                        'Date de génération': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                    })
                    summary.to_excel(writer, sheet_name='Récapitulatif', index=False)
                    
                print(f"✅ Budget Excel généré : {filename}")
                return filename
            except Exception as e:
                print(f"⚠️ Erreur lors de la génération Excel: {e}")
                print("   Utilisation du format CSV à la place.")
        
        # Fallback: version CSV
        csv_filename = base_filename + ".csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"✅ Budget CSV généré : {csv_filename}")
        
        # Générer aussi un fichier récapitulatif
        summary_filename = base_filename + "_summary.csv"
        summary_df = pd.DataFrame({
            'Projet': [self.project_name],
            'Catégorie': [self.category],
            'Budget total (FCFA)': [self.total_budget],
            'Budget calculé (FCFA)': [grand_total],
            'Écart': [self.total_budget - grand_total],
            'Date': [datetime.now().strftime('%Y-%m-%d')]
        })
        summary_df.to_csv(summary_filename, index=False, encoding='utf-8')
        
        return csv_filename


# ============================================
# PROJET 1 - SIAP-GAT (Catégorie A - 150 M FCFA)
# ============================================
print("\n" + "🔥"*30)
print(" GÉNÉRATION DES BUDGETS POUR LES 4 PROJETS ")
print("🔥"*30)

p1 = ProjectBudget(
    project_name="SIAP-GAT - Système Intégré d'Alerte Précoce et de Gestion Agroécologique",
    category="A - Solutions-Pays",
    total_budget=150_000_000,
    duration_months=36
)

print("\n📌 PROJET 1: SIAP-GAT (150 M FCFA)")

# ÉQUIPEMENTS (30% - 45 M FCFA)
p1.add_line('Équipements', 'Capteurs IoT', 'Capteurs humidité sol (500 unités)', 45000, 500)
p1.add_line('Équipements', 'Passerelles LoRa', 'Passerelles LoRaWAN (10 unités)', 350000, 10)
p1.add_line('Équipements', 'Serveurs', 'Serveurs de calcul et stockage', 5000000, 2)
p1.add_line('Équipements', 'Drones', 'Drones avec capteurs multispectraux', 8000000, 2)
p1.add_line('Équipements', 'Véhicule', 'Véhicule tout-terrain pour missions terrain', 12000000, 1)
p1.add_line('Équipements', 'Matériel informatique', 'Ordinateurs portables (10 unités)', 500000, 10)
p1.add_line('Équipements', 'Tablettes', 'Tablettes pour agents de terrain (50 unités)', 150000, 50)

# FONCTIONNEMENT (40% - 60 M FCFA)
p1.add_line('Fonctionnement', 'Missions terrain', 'Per diem missions (500 jours)', 25000, 500)
p1.add_line('Fonctionnement', 'Carburant', 'Carburant pour véhicule et drones', 2000000, 1, 36)
p1.add_line('Fonctionnement', 'Consommables', 'Kits d\'entretien capteurs', 500000, 1, 36)
p1.add_line('Fonctionnement', 'Communication', 'Frais de connectivité (équipe)', 300000, 1, 36)
p1.add_line('Fonctionnement', 'Logiciels', 'Licences et abonnements cloud', 2000000, 1, 36)

# PERSONNEL (20% - 30 M FCFA)
p1.add_line('Personnel', 'Post-doctorant', 'Bourse post-doctorale (1 x 36 mois)', 350000, 1, 36)
p1.add_line('Personnel', 'Doctorants', 'Bourses doctorales (4 x 36 mois)', 200000, 4, 36)
p1.add_line('Personnel', 'Masters', 'Bourses master (8 x 12 mois par an)', 100000, 8, 36)
p1.add_line('Personnel', 'Indemnités PER', 'Indemnités chercheurs seniors (3 x 36 mois)', 150000, 3, 36)

# ANIMATION ET VALORISATION (10% - 15 M FCFA)
p1.add_line('Animation', 'Ateliers', 'Ateliers de concertation (10 ateliers)', 500000, 10)
p1.add_line('Animation', 'Formations', 'Formation agents numériques (1000 pers.)', 5000, 1000)
p1.add_line('Animation', 'Publications', 'Frais de publication (6 articles)', 500000, 6)
p1.add_line('Animation', 'Traductions', 'Traduction en langues nationales', 2000000, 1)
p1.add_line('Animation', 'Communication', 'Productions audiovisuelles', 3000000, 1)

p1.display_summary()
p1.generate_excel()

# ============================================
# PROJET 2 - OPSOL-AI (Catégorie B - 50 M FCFA)
# ============================================
p2 = ProjectBudget(
    project_name="OPSOL-AI - Observatoire Participatif de la Santé des Sols",
    category="B - Écosystèmes",
    total_budget=50_000_000,
    duration_months=24
)

print("\n📌 PROJET 2: OPSOL-AI (50 M FCFA)")

# ÉQUIPEMENTS (30% - 15 M FCFA)
p2.add_line('Équipements', 'Kits analyse', 'Kits d\'analyse de sols (50 kits)', 150000, 50)
p2.add_line('Équipements', 'GPS', 'GPS de terrain (10 unités)', 200000, 10)
p2.add_line('Équipements', 'Tablettes', 'Tablettes pour sentinelles (20 unités)', 150000, 20)
p2.add_line('Équipements', 'Matériel labo', 'Petit matériel de laboratoire', 2000000, 1)

# FONCTIONNEMENT (40% - 20 M FCFA)
p2.add_line('Fonctionnement', 'Missions terrain', 'Per diem missions (300 jours)', 25000, 300)
p2.add_line('Fonctionnement', 'Carburant', 'Carburant', 500000, 1, 24)
p2.add_line('Fonctionnement', 'Analyses labo', 'Analyses de sol de référence', 50000, 200)
p2.add_line('Fonctionnement', 'Consommables', 'Consommables terrain', 300000, 1, 24)

# PERSONNEL (20% - 10 M FCFA)
p2.add_line('Personnel', 'Doctorants', 'Bourses doctorales (2 x 24 mois)', 200000, 2, 24)
p2.add_line('Personnel', 'Masters', 'Bourses master (4 x 12 mois)', 100000, 4, 12)
p2.add_line('Personnel', 'Indemnités PER', 'Indemnités chercheurs (3 x 24 mois)', 100000, 3, 24)

# ANIMATION (10% - 5 M FCFA)
p2.add_line('Animation', 'Ateliers', 'Ateliers participatifs (6 ateliers)', 400000, 6)
p2.add_line('Animation', 'Formations', 'Formation sentinelles (300 pers.)', 5000, 300)
p2.add_line('Animation', 'Publications', 'Frais de publication (4 articles)', 400000, 4)

p2.display_summary()
p2.generate_excel()

# ============================================
# PROJET 3 - PUBLI-AGRO (Catégorie B - 50 M FCFA)
# ============================================
p3 = ProjectBudget(
    project_name="PUBLI-AGRO - Plateforme Multilingue de Publication Scientifique",
    category="B - Écosystèmes",
    total_budget=50_000_000,
    duration_months=24
)

print("\n📌 PROJET 3: PUBLI-AGRO (50 M FCFA)")

# ÉQUIPEMENTS (20% - 10 M FCFA)
p3.add_line('Équipements', 'Studio', 'Studio d\'enregistrement', 5000000, 1)
p3.add_line('Équipements', 'Informatique', 'Ordinateurs et serveurs', 3000000, 1)
p3.add_line('Équipements', 'Enregistreurs', 'Enregistreurs audio (10 unités)', 200000, 10)

# FONCTIONNEMENT (40% - 20 M FCFA)
p3.add_line('Fonctionnement', 'Production', 'Production contenus (50 articles)', 200000, 50)
p3.add_line('Fonctionnement', 'Radios', 'Partenariats radios communautaires', 300000, 20)
p3.add_line('Fonctionnement', 'Missions', 'Missions dans les régions', 200000, 20)
p3.add_line('Fonctionnement', 'Impression', 'Impression guides et fiches', 2000000, 1)

# PERSONNEL (30% - 15 M FCFA)
p3.add_line('Personnel', 'Doctorants', 'Bourses doctorales (2 x 24 mois)', 200000, 2, 24)
p3.add_line('Personnel', 'Masters', 'Bourses master (4 x 12 mois)', 100000, 4, 12)
p3.add_line('Personnel', 'Traducteurs', 'Honoraires traducteurs (30 pers.)', 200000, 30)
p3.add_line('Personnel', 'Journalistes', 'Honoraires journalistes', 150000, 12, 12)

# ANIMATION (10% - 5 M FCFA)
p3.add_line('Animation', 'Ateliers', 'Ateliers de formation traducteurs', 500000, 4)
p3.add_line('Animation', 'Séminaires', 'Séminaires de vulgarisation', 500000, 4)
p3.add_line('Animation', 'Valorisation', 'Promotion de la plateforme', 1000000, 1)

p3.display_summary()
p3.generate_excel()

# ============================================
# PROJET 4 - IA-SEMENCE (Catégorie C - 25 M FCFA)
# ============================================
p4 = ProjectBudget(
    project_name="IA-SEMENCE - Gestion Communautaire des Semences par IA",
    category="C - Pépinières",
    total_budget=25_000_000,
    duration_months=12
)

print("\n📌 PROJET 4: IA-SEMENCE (25 M FCFA)")

# ÉQUIPEMENTS (28% - 7 M FCFA)
p4.add_line('Équipements', 'Tablettes', 'Tablettes pour banques semences (15 unités)', 150000, 15)
p4.add_line('Équipements', 'Serveur', 'Serveur applicatif', 2500000, 1)
p4.add_line('Équipements', 'GPS', 'GPS (5 unités)', 200000, 5)

# FONCTIONNEMENT (40% - 10 M FCFA)
p4.add_line('Fonctionnement', 'Missions terrain', 'Per diem missions (150 jours)', 25000, 150)
p4.add_line('Fonctionnement', 'Carburant', 'Carburant', 300000, 1, 12)
p4.add_line('Fonctionnement', 'Développement', 'Développement application', 2000000, 1)

# PERSONNEL (20% - 5 M FCFA)
p4.add_line('Personnel', 'Doctorant', 'Bourse doctorale (1 x 12 mois)', 200000, 1, 12)
p4.add_line('Personnel', 'Masters', 'Bourses master (2 x 6 mois)', 100000, 2, 6)
p4.add_line('Personnel', 'Indemnités PER', 'Indemnités chercheurs (3 x 12 mois)', 80000, 3, 12)

# ANIMATION (12% - 3 M FCFA)
p4.add_line('Animation', 'Formations', 'Formation gestionnaires (50 pers.)', 30000, 50)
p4.add_line('Animation', 'Ateliers', 'Ateliers participatifs (4 ateliers)', 300000, 4)
p4.add_line('Animation', 'Publications', 'Frais de publication (2 articles)', 300000, 2)

p4.display_summary()
p4.generate_excel()

print("\n" + "✅"*20)
print("✅ GÉNÉRATION DES BUDGETS TERMINÉE ✅")
print("✅"*20)
print(f"\n📁 Tous les budgets ont été générés dans le dossier: {OUTPUT_DIR}/")
print("\nFichiers créés :")
for file in os.listdir(OUTPUT_DIR):
    print(f"  - {file}")

print("\n" + "="*60)
print("RÉCAPITULATIF DES BUDGETS PAR PROJET")
print("="*60)
print(f"{'Projet':<30} {'Budget (FCFA)':>15} {'Catégorie':>15}")
print("-"*60)
print(f"{'SIAP-GAT':<30} {150_000_000:>15,} {'A':>15}")
print(f"{'OPSOL-AI':<30} {50_000_000:>15,} {'B':>15}")
print(f"{'PUBLI-AGRO':<30} {50_000_000:>15,} {'B':>15}")
print(f"{'IA-SEMENCE':<30} {25_000_000:>15,} {'C':>15}")
print("-"*60)
print(f"{'TOTAL':<30} {275_000_000:>15,} {'':>15}")
print("="*60)