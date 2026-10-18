"""Petits outils de l'analyse dans le temps.

RÔLE      : fournir quatre briques simples, combinées ensuite par `AnalyseurTemporel`.
ENTRÉE    : des valeurs ou des événements horodatés (en secondes).
SORTIE    : valeurs lissées, fractions de temps, nombres d'événements, durées.
PIPELINE  : bloc 7 (analyse temporelle).

POURQUOI  : une seule image ne permet pas de conclure (un œil fermé est presque toujours
            un clignement). La somnolence se lit dans la DURÉE et la RÉPÉTITION.
"""

from __future__ import annotations

from collections import deque
from itertools import islice
from typing import Deque, Optional, Tuple


class LisseurExponentiel:
    """Lissage par moyenne mobile exponentielle (EMA).

    FORMULE : s(t) = α × x(t) + (1 − α) × s(t − 1)
    Exemple (α = 0,4, s précédent = 0,30, x = 0,10) : s = 0,04 + 0,18 = 0,22.
    """

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._valeur_lissee: Optional[float] = None

    def lisser(self, valeur: float) -> float:
        """Renvoie la valeur lissée.

        POURQUOI : empêcher qu'une seule image aberrante ne déclenche ou n'interrompe une
        fermeture. Le filtre est récursif : il ne mémorise qu'une seule valeur.
        """
        if self._valeur_lissee is None:
            self._valeur_lissee = valeur
        else:
            self._valeur_lissee = self._alpha * valeur + (1.0 - self._alpha) * self._valeur_lissee
        return self._valeur_lissee

    def reinitialiser(self) -> None:
        """Oublie l'historique (utile après une perte du visage)."""
        self._valeur_lissee = None


class FenetrePerclos:
    """Fenêtre glissante sur l'état « œil fermé », pour calculer le PERCLOS.

    PERCLOS = (temps passé yeux fermés dans la fenêtre) / (durée de la fenêtre, 60 s).
    On divise par la durée COMPLÈTE de la fenêtre : juste après le démarrage, quelques
    secondes de fermeture ne donnent donc pas un PERCLOS artificiellement élevé.
    """

    def __init__(self, duree: float) -> None:
        self._duree = duree
        # APPORT : `deque` (file à deux bouts) permet de retirer les plus anciens éléments
        # en tête de file très rapidement, ce qui convient à une fenêtre glissante.
        self._observations: Deque[Tuple[float, bool]] = deque()

    def ajouter(self, instant: float, ferme: bool) -> None:
        """Ajoute une observation (instant, œil fermé ?) et retire les plus anciennes."""
        self._observations.append((instant, ferme))
        self._purger(instant)

    def valeur(self) -> float:
        """Renvoie la fraction de temps « fermé » sur la fenêtre (0 à 1) : le PERCLOS.

        Chaque observation « fermé » compte pour l'intervalle de temps qui la sépare de la
        suivante : le résultat ne dépend donc pas de la cadence de la caméra.
        """
        if len(self._observations) < 2:
            return 0.0
        temps_ferme = 0.0
        precedente = self._observations[0]
        for suivante in islice(self._observations, 1, None):
            if precedente[1]:
                temps_ferme += suivante[0] - precedente[0]
            precedente = suivante
        return min(1.0, temps_ferme / self._duree)

    def _purger(self, instant: float) -> None:
        """Retire les observations plus vieilles que la durée de la fenêtre.

        POURQUOI : garder une mémoire bornée et une fenêtre réellement glissante.
        """
        while self._observations and instant - self._observations[0][0] > self._duree:
            self._observations.popleft()


class CompteurEvenements:
    """Compte les événements survenus récemment (fermetures longues, bâillements)."""

    def __init__(self, duree: float) -> None:
        self._duree = duree
        self._instants: Deque[float] = deque()

    def enregistrer(self, instant: float) -> None:
        """Ajoute un événement en mémorisant son instant."""
        self._instants.append(instant)

    def nombre(self, instant: float) -> int:
        """Renvoie le nombre d'événements dans la fenêtre (un événement isolé ne dit rien,
        c'est la répétition qui signale la fatigue)."""
        self._purger(instant)
        return len(self._instants)

    def _purger(self, instant: float) -> None:
        """Retire les événements trop anciens."""
        while self._instants and instant - self._instants[0] > self._duree:
            self._instants.popleft()


class ChronometreEtat:
    """Mesure la durée continue pendant laquelle une condition reste vraie.

    POURQUOI : distinguer un clignement (0,1 à 0,4 s) d'un endormissement (1,5 s et plus).
    """

    def __init__(self) -> None:
        self._instant_debut: Optional[float] = None

    def mettre_a_jour(self, instant: float, actif: bool) -> Tuple[float, Optional[float]]:
        """Renvoie (durée en cours, durée terminée ou None).

        - tant que la condition est vraie : durée en cours > 0 ;
        - à l'instant où elle devient fausse : la durée de la période qui vient de finir.
        POURQUOI : reconnaître à la fois le début et la fin d'une fermeture.
        """
        if actif:
            if self._instant_debut is None:
                self._instant_debut = instant
            return instant - self._instant_debut, None
        if self._instant_debut is not None:
            duree_terminee = instant - self._instant_debut
            self._instant_debut = None
            return 0.0, duree_terminee
        return 0.0, None

    def reinitialiser(self) -> None:
        """Remet le chronomètre à zéro (évite une « durée fantôme »)."""
        self._instant_debut = None
