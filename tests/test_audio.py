"""Tests de la génération de la sonnerie (série de bips courts).

Lancer avec :  pytest tests/        (depuis la racine du projet)
"""

from __future__ import annotations

import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alerte.generation_audio import GenerateurSonnerie


def test_duree_de_la_sonnerie() -> None:
    generateur = GenerateurSonnerie()
    signal = generateur._synthetiser()
    attendu = (generateur.NOMBRE_BIPS * generateur.DUREE_BIP
               + (generateur.NOMBRE_BIPS - 1) * generateur.PAUSE_ENTRE_BIPS)
    duree = len(signal) / generateur.FREQUENCE_ECHANTILLONNAGE
    assert abs(duree - attendu) < 0.01
    assert duree < 5.0                      # la sonnerie reste courte (moins de 5 secondes)


def test_la_sonnerie_contient_le_bon_nombre_de_bips() -> None:
    generateur = GenerateurSonnerie()
    signal = np.abs(generateur._synthetiser().astype(float))
    actif = signal > 0.05 * signal.max()    # 1 quand le son est audible
    # Compte les passages du silence au son : un passage = le début d'un bip.
    # Les bips sont séparés par des silences assez longs pour que le comptage soit fiable
    # à condition de lisser le signal sur quelques millisecondes.
    noyau = np.ones(200) / 200
    enveloppe = np.convolve(signal, noyau, mode="same")
    audible = enveloppe > 0.1 * enveloppe.max()
    debuts = int(np.sum(audible[1:] & ~audible[:-1])) + int(audible[0])
    assert debuts == generateur.NOMBRE_BIPS


def test_pas_de_claquement_au_debut_et_a_la_fin() -> None:
    signal = GenerateurSonnerie()._synthetiser()
    assert abs(int(signal[0])) < 500 and abs(int(signal[-1])) < 500   # démarre et finit près de zéro


def test_ecriture_du_fichier_wav() -> None:
    with tempfile.TemporaryDirectory() as dossier:
        chemin = Path(dossier) / "sonnerie.wav"
        GenerateurSonnerie().generer(chemin)
        with wave.open(str(chemin), "rb") as fichier:
            assert fichier.getnchannels() == 1
            assert fichier.getsampwidth() == 2
            assert fichier.getframerate() == 44100
            assert fichier.getnframes() > 0
