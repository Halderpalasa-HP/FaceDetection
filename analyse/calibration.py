"""Calibration : mesurer l'EAR « yeux ouverts » de l'utilisateur pour fixer son seuil.

RÔLE      : personnaliser le système (un EAR de 0,22 est normal pour une personne aux yeux
            étroits mais signale une fermeture chez une autre).
ENTRÉE    : pendant 5 secondes, l'EAR moyen et le tangage de chaque image.
SORTIE    : un `ProfilUtilisateur` (seuil d'EAR et tangage de référence).
PIPELINE  : bloc 6 (calibration).

FORMULES  : seuil_EAR = ratio × médiane(EAR mesurés)        avec ratio = 0,70
            tangage_référence = médiane(tangages mesurés)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from config import Configuration
from noyau.modeles import ProfilUtilisateur


class Calibrateur:
    """Collecte des mesures pendant quelques secondes puis construit le profil."""

    def __init__(self, config: Configuration) -> None:
        self._config = config
        self._mesures_ear: List[float] = []
        self._mesures_tangage: List[float] = []
        self._instant_debut: Optional[float] = None

    def ajouter_mesure(self, instant: float, ear: float, tangage: float) -> None:
        """Mémorise une mesure (le chronomètre démarre à la première mesure reçue).

        POURQUOI : fournir la matière de la médiane. Si aucun visage n'est détecté, aucune
        mesure n'arrive et le chronomètre ne démarre pas.
        """
        if self._instant_debut is None:
            self._instant_debut = instant
        self._mesures_ear.append(ear)
        self._mesures_tangage.append(tangage)

    def progression(self, instant: float) -> float:
        """Renvoie l'avancement de la calibration, de 0 (début) à 1 (terminée).

        POURQUOI : alimenter la barre de progression affichée à l'écran.
        """
        if self._instant_debut is None:
            return 0.0
        ecoule = instant - self._instant_debut
        return max(0.0, min(1.0, ecoule / self._config.duree_calibration))

    def terminee(self, instant: float) -> bool:
        """Indique si la durée de calibration est écoulée.

        POURQUOI : décider du passage de la phase de calibration à la surveillance.
        """
        return self.progression(instant) >= 1.0

    def construire_profil(self) -> Optional[ProfilUtilisateur]:
        """Calcule le profil, ou renvoie `None` (et repart de zéro) si la calibration est faussée.

        POURQUOI : transformer les mesures en un seuil propre à la personne.
        """
        if not self._mesures_ear:
            return None

        # APPORT : la MÉDIANE (valeur du milieu des mesures triées) ignore les clignements,
        # qui font chuter brièvement l'EAR. La moyenne, elle, serait tirée vers le bas.
        mediane_ear = float(np.median(self._mesures_ear))
        mediane_tangage = float(np.median(self._mesures_tangage))

        if not self._est_valide(mediane_ear):
            # Calibration faussée (yeux mi-clos, reflets de lunettes…) : on recommence.
            self._mesures_ear.clear()
            self._mesures_tangage.clear()
            self._instant_debut = None
            return None

        return ProfilUtilisateur(
            seuil_ear=self._config.ratio_seuil_ear * mediane_ear,
            tangage_reference=mediane_tangage,
        )

    def _est_valide(self, mediane_ear: float) -> bool:
        """Rejette une médiane anormalement basse (inférieure à 0,15 par défaut).

        POURQUOI : un EAR médian trop faible indique que les yeux étaient presque fermés
        pendant la calibration ; le seuil obtenu serait inutilisable.
        """
        return mediane_ear >= self._config.ear_minimal_valide
