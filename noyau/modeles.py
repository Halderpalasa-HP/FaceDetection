"""Objets de données partagés entre tous les modules.

RÔLE      : définir les « paquets de données » qui circulent dans le pipeline.
ENTRÉE    : aucune (simples structures).
SORTIE    : classes `Etat`, `ProfilUtilisateur`, `Indicateurs`, `ResultatVisage`,
            `OptionsAffichage`, `SortieTraitement`.
PIPELINE  : transversal. Chaque bloc reçoit et renvoie l'un de ces objets.

POURQUOI  : un objet nommé et typé est plus lisible et plus sûr qu'un tuple ou qu'une
            liste dont il faut deviner le contenu.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

# Un point du visage en pixels : (x, y).
Point = Tuple[float, float]


class Etat(Enum):
    """Les cinq états possibles du système.

    APPORT : `Enum` remplace des chaînes de caractères fragiles (« endormi », « Endormi »…)
             par des valeurs uniques ; une faute de frappe devient une erreur immédiate.
    """

    CALIBRATION = "CALIBRATION"
    EVEILLE = "ÉVEILLÉ"
    SOMNOLENT = "SOMNOLENT"
    ENDORMI = "ENDORMI"
    VISAGE_PERDU = "VISAGE PERDU"

    @property
    def libelle(self) -> str:
        """Texte français affiché à l'écran.

        POURQUOI : séparer ce que l'on affiche (le libellé) de ce que l'on compare (l'état).
        """
        return self.value


@dataclass
class ProfilUtilisateur:
    """Résultat de la calibration : ce qui rend le système propre à la personne."""

    seuil_ear: float          # en dessous de ce EAR lissé, l'œil est considéré fermé
    tangage_reference: float  # tangage (degrés) de la tête au repos


@dataclass
class Indicateurs:
    """Sorties de l'analyse temporelle, entrées de la décision."""

    ferme: bool                 # l'œil est-il fermé à cet instant ?
    duree_fermeture: float      # durée (s) de la fermeture continue en cours
    perclos: float              # fraction du temps yeux fermés sur la fenêtre (0 à 1)
    fermetures_longues: int     # nombre de fermetures de 0,5 à 1,5 s récentes
    baillements: int            # nombre de bâillements récents
    ear_lisse: float            # EAR après lissage exponentiel


@dataclass
class ResultatVisage:
    """Les points du visage détectés, convertis en pixels.

    `coordonnees` contient 478 points (468 du visage + 10 d'iris) sous la forme (x, y).
    """

    coordonnees: List[Point]
    largeur: int
    hauteur: int

    def point(self, indice: int) -> Point:
        """Renvoie le point (x, y) d'indice donné.

        POURQUOI : point d'accès unique aux indices MediaPipe, plus facile à relire.
        """
        return self.coordonnees[indice]

    def points(self, indices: Sequence[int]) -> List[Point]:
        """Renvoie la liste des points demandés (par exemple les 6 points d'un œil).

        POURQUOI : extraire les 6 points d'un œil en une seule ligne.
        """
        return [self.coordonnees[i] for i in indices]

    def en_entiers(self) -> List[Tuple[int, int]]:
        """Renvoie tous les points arrondis à l'entier le plus proche.

        POURQUOI : les fonctions de dessin d'OpenCV (`cv2.line`) n'acceptent que des entiers.
        """
        return [(int(round(x)), int(round(y))) for x, y in self.coordonnees]


@dataclass
class OptionsAffichage:
    """Interrupteurs de l'interface : MAILLAGE, GUIDES, VALEURS."""

    maillage: bool = True
    guides: bool = True
    valeurs: bool = True

    def basculer(self, nom: str) -> None:
        """Inverse l'option nommée (`maillage`, `guides` ou `valeurs`).

        POURQUOI : utilisée à la fois par la souris et par le clavier.
        """
        if not hasattr(self, nom):
            raise ValueError(f"Option d'affichage inconnue : {nom}")
        setattr(self, nom, not getattr(self, nom))


@dataclass
class SortieTraitement:
    """Tout ce qui sort du traitement d'une image : un seul paquet pour l'interface."""

    etat: Etat
    visage: Optional[ResultatVisage] = None
    indicateurs: Optional[Indicateurs] = None
    ear: float = 0.0                 # EAR brut (moyenne des deux yeux)
    mar: float = 0.0                 # ouverture de la bouche
    ecart_tangage: float = 0.0       # écart (degrés) par rapport au tangage de référence
    seuil_ear: float = 0.0           # seuil personnel issu de la calibration
    progression: float = 0.0         # avancement de la calibration (0 à 1)
    instant: float = 0.0             # horodatage (secondes) de l'image traitée
