"""Application : orchestre la caméra, le pipeline, l'alerte et l'interface.

RÔLE      : relier tous les modules et exécuter la boucle temps réel.
ENTRÉE    : aucune (utilise la configuration, la caméra et le clavier/souris).
SORTIE    : une fenêtre clinique en direct et, si nécessaire, la séquence d'alerte sonore.
PIPELINE  : chef d'orchestre des blocs 1 à 9.

POURQUOI  : c'est la seule classe qui connaît tous les modules ; chacun des autres reste
            indépendant et testable séparément.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from alerte.gestionnaire_alerte import GestionnaireAlerte
from config import Configuration
from interface.liaisons import LiaisonsVisage
from interface.rendu_masque import RenduMasque
from interface.tableau_de_bord import TableauDeBord
from noyau.camera import Camera
from noyau.modeles import Etat, SortieTraitement
from noyau.suivi_visage import SuiviVisage
from pipeline import PipelineSurveillance

NOM_FENETRE = "Detection de somnolence"


class Application:
    """Construit tous les modules puis lance la boucle principale."""

    def __init__(self, config: Optional[Configuration] = None) -> None:
        self._config = config or Configuration.par_defaut()
        self._config.valider()
        self._camera = Camera(self._config.indice_camera, self._config.largeur_image,
                              self._config.hauteur_image)
        self._suivi = SuiviVisage(self._config)
        self._pipeline = PipelineSurveillance(self._config, self._suivi)
        self._alerte = GestionnaireAlerte(self._config)
        self._tableau = TableauDeBord(self._config, RenduMasque(LiaisonsVisage()))

    def lancer(self) -> None:
        """Ouvre les ressources, exécute la boucle, puis libère tout proprement.

        POURQUOI : appelée par `main.py`. Le bloc `finally` garantit la libération de la
        caméra et du son même en cas d'erreur ou d'interruption.
        """
        try:
            self._camera.ouvrir()
            cv2.namedWindow(NOM_FENETRE, cv2.WINDOW_AUTOSIZE)
            # APPORT : `cv2.setMouseCallback` relie les clics de la souris à notre méthode.
            cv2.setMouseCallback(NOM_FENETRE, self._tableau.gerer_souris)
            self._boucle()
        finally:
            self._nettoyer()

    def _boucle(self) -> None:
        """Cycle temps réel : lire, traiter, alerter, afficher, écouter le clavier.

        POURQUOI : c'est le rythme du système ; chaque tour traite une image de la caméra.
        """
        fps = 0.0
        instant_precedent = time.monotonic()
        while True:
            # Démarrage ou arrêt de la caméra demandé depuis l'interface.
            if self._tableau.en_marche and not self._camera.est_ouverte():
                self._camera.ouvrir()
                self._pipeline = PipelineSurveillance(self._config, self._suivi)   # recalibration
            if not self._tableau.en_marche and self._camera.est_ouverte():
                self._camera.fermer()
                self._alerte.arreter()

            if self._camera.est_ouverte():
                lecture = self._camera.lire()
                if lecture is None:
                    print("Erreur : la caméra ne fournit plus d'image.")
                    break
                image, instant = lecture
                sortie = self._pipeline.traiter(image, instant)
                self._gerer_alerte(sortie.etat)
                fps = 0.9 * fps + 0.1 / max(instant - instant_precedent, 1e-6)
                instant_precedent = instant
            else:
                # Caméra arrêtée : écran d'attente (image noire).
                image = np.zeros((self._config.hauteur_image, self._config.largeur_image, 3), np.uint8)
                sortie = SortieTraitement(etat=Etat.EVEILLE, instant=time.monotonic())
                fps = 0.0

            cadre = self._tableau.composer(image, sortie, self._alerte.statut(), fps)
            cv2.imshow(NOM_FENETRE, cadre)

            touche = cv2.waitKey(1) & 0xFF
            if self._tableau.gerer_touche(touche) or self._tableau.quitter_demande:
                break
            # Fermeture par la croix de la fenêtre.
            if cv2.getWindowProperty(NOM_FENETRE, cv2.WND_PROP_VISIBLE) < 1:
                break

    def _gerer_alerte(self, etat: Etat) -> None:
        """Relie la décision à l'action : ENDORMI déclenche l'alerte, SOMNOLENT fait un bip.

        POURQUOI : dans tous les autres cas, l'alerte en cours est interrompue : elle s'arrête
        dès que le conducteur n'est plus dans l'état ENDORMI (par exemple yeux rouverts).
        """
        if etat is Etat.ENDORMI:
            self._alerte.declencher()
        else:
            self._alerte.arreter()
            if etat is Etat.SOMNOLENT:
                self._alerte.bip()

    def _nettoyer(self) -> None:
        """Libère la caméra, le modèle, les sons et la fenêtre (fermeture propre)."""
        self._alerte.fermer()
        self._suivi.fermer()
        self._camera.fermer()
        cv2.destroyAllWindows()
