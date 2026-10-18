"""Point d'entrée du programme : `python main.py`.

RÔLE      : construire l'application et la lancer, avec des messages d'erreur clairs en français.
PIPELINE  : démarrage.
"""

from __future__ import annotations

import sys

from app import Application


def main() -> int:
    """Lance l'application ; renvoie 0 si tout va bien, 1 en cas d'erreur de démarrage."""
    try:
        Application().lancer()
    except (FileNotFoundError, RuntimeError, ValueError) as erreur:
        # Fichier modèle ou son absent, caméra introuvable, réglage incohérent…
        print(f"Erreur : {erreur}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
