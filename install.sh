#!/bin/bash

# --- Configuration ---
PLUGIN_NAME="ArriGrainOFX.ofx.bundle"
DEST_DIR="/Library/OFX/Plugins"

echo "=========================================="
echo "  ArriGrain OFX Plugin Installer"
echo "=========================================="

# Se placer dans le dossier où se trouve le script
cd "$(dirname "$0")"

# Vérifier que le plugin est bien dans le même dossier que le script
if [ ! -d "$PLUGIN_NAME" ]; then
    echo "❌ Erreur : Le fichier $PLUGIN_NAME est introuvable."
    echo "Assurez-vous que ce script est dans le même dossier décompressé que le plugin."
    exit 1
fi

# Demander les droits administrateur (requis pour écrire dans /Library et utiliser xattr)
echo "⚠️  L'installation nécessite les droits administrateur."
echo "Veuillez entrer votre mot de passe de session Mac :"
sudo -v

# 1. Créer le dossier de destination s'il n'existe pas
sudo mkdir -p "$DEST_DIR"

# 2. Supprimer l'ancienne version s'il y en a une, puis copier la nouvelle
echo "📂 Copie de $PLUGIN_NAME vers $DEST_DIR..."
sudo rm -rf "$DEST_DIR/$PLUGIN_NAME"
sudo cp -r "$PLUGIN_NAME" "$DEST_DIR/"

# 3. Nettoyer les attributs de quarantaine (Gatekeeper)
echo "🔓 Suppression de la quarantaine Apple (Gatekeeper)..."
sudo xattr -rd com.apple.quarantine "$DEST_DIR/$PLUGIN_NAME" 2>/dev/null

# 4. Forcer la signature locale ad-hoc
echo "✍️  Application de la signature locale (Ad-hoc)..."
sudo codesign -f -s - "$DEST_DIR/$PLUGIN_NAME"

echo "=========================================="
echo "✅ Installation terminée avec succès !"
echo "Vous pouvez maintenant lancer DaVinci Resolve."
echo "=========================================="
