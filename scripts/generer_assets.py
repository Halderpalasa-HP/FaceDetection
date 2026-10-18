"""Génère les fichiers audio de l'alerte : `assets/sonnerie.wav` et `assets/message.wav`.

À exécuter une seule fois, depuis la racine du projet :  python scripts/generer_assets.py
Le modèle `face_landmarker.task` doit être téléchargé séparément (voir README.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permet d'importer les modules du projet quand on lance le script depuis le dossier « scripts ».
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alerte.generation_audio import GenerateurSonnerie, GenerateurVoix
from config import Configuration


def main() -> None:
    """Crée la sonnerie (cloche synthétisée) puis le message vocal (voix française)."""
    config = Configuration.par_defaut()
    print("Génération de la sonnerie…")
    GenerateurSonnerie().generer(config.chemin_sonnerie)
    print(f"  -> {config.chemin_sonnerie}")
    print("Génération du message vocal…")
    GenerateurVoix().generer(config.chemin_voix, config.message_vocal)
    print(f"  -> {config.chemin_voix}")
    print("Terminé.")


if __name__ == "__main__":
    main()
