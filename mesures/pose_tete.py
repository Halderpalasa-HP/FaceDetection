"""Estimation de la pose de la tête : tangage, lacet et roulis.

RÔLE      : savoir si la tête est penchée vers l'avant (signe d'endormissement).
ENTRÉE    : un `ResultatVisage` (points 2D en pixels).
SORTIE    : trois angles en degrés (tangage, lacet, roulis) ou `None` si le calcul échoue.
PIPELINE  : bloc 5 (mesures).

PRINCIPE (problème « PnP », Perspective-n-Point) :
  On connaît la position 3D de 6 points sur un visage générique, et leur position 2D dans
  l'image. On cherche la rotation R et la translation t qui projettent le modèle 3D sur ces
  points, selon l'équation :   s · [u, v, 1]ᵀ = K · [ R | t ] · [X, Y, Z, 1]ᵀ
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from noyau.modeles import ResultatVisage


class EstimateurPose:
    """Estime l'orientation 3D de la tête à partir de six points du visage."""

    # Indices MediaPipe : bout du nez, menton, coin externe de chaque œil, coins de la bouche.
    INDICES_POSE = (1, 152, 33, 263, 61, 291)

    def __init__(self) -> None:
        # Modèle 3D générique du visage (unités arbitraires), dans le même ordre que les indices.
        self._modele_3d = np.array(
            [
                (0.0, 0.0, 0.0),           # bout du nez
                (0.0, -330.0, -65.0),      # menton
                (-225.0, 170.0, -135.0),   # coin externe œil (côté gauche de l'image)
                (225.0, 170.0, -135.0),    # coin externe œil (côté droit de l'image)
                (-150.0, -150.0, -125.0),  # coin de bouche (côté gauche de l'image)
                (150.0, -150.0, -125.0),   # coin de bouche (côté droit de l'image)
            ],
            dtype=np.float64,
        )

    def estimer(self, visage: ResultatVisage) -> Optional[Tuple[float, float, float]]:
        """Renvoie (tangage, lacet, roulis) en degrés, ou `None` si la résolution échoue.

        POURQUOI : une tête qui tombe vers l'avant confirme un endormissement, même quand
        les yeux sont encore partiellement ouverts.
        """
        points_image = np.array(visage.points(self.INDICES_POSE), dtype=np.float64)
        matrice_camera = self._matrice_camera(visage.largeur, visage.hauteur)

        # APPORT : `cv2.solvePnP` retrouve la rotation 3D de la tête en minimisant l'erreur de
        # reprojection entre les points 2D observés et les points du modèle 3D projetés.
        succes, vecteur_rotation, _translation = cv2.solvePnP(
            self._modele_3d, points_image, matrice_camera, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not succes:
            return None

        # APPORT : `cv2.Rodrigues` convertit le vecteur de rotation (axe × angle) en matrice
        # de rotation 3 × 3 grâce à la formule de Rodrigues.
        matrice_rotation, _ = cv2.Rodrigues(vecteur_rotation)
        return self._angles_depuis_rotation(matrice_rotation)

    def _matrice_camera(self, largeur: int, hauteur: int) -> np.ndarray:
        """Construit la matrice intrinsèque K de la caméra (approximation).

        K = | f  0  cx |   avec f ≈ largeur de l'image et (cx, cy) = centre de l'image.
            | 0  f  cy |
            | 0  0   1 |

        POURQUOI : `solvePnP` exige cette matrice ; sans calibration de la caméra, la
        focale est approchée par la largeur de l'image, ce qui suffit pour un prototype.
        """
        focale = float(largeur)
        return np.array(
            [[focale, 0.0, largeur / 2.0],
             [0.0, focale, hauteur / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def _angles_depuis_rotation(self, rotation: np.ndarray) -> Tuple[float, float, float]:
        """Convertit la matrice de rotation en angles d'Euler (degrés).

        tangage = atan2(R32, R33)
        lacet   = atan2(−R31, √(R32² + R33²))
        roulis  = atan2(R21, R11)

        POURQUOI : des angles en degrés sont lisibles par un humain, contrairement à une matrice.
        """
        tangage = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
        lacet = math.degrees(
            math.atan2(-rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2]))
        )
        roulis = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        return tangage, lacet, roulis
