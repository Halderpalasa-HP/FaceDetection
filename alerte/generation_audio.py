"""Génération des fichiers audio de l'alerte (à faire une seule fois).

RÔLE      : fabriquer `sonnerie.wav` (série de bips courts « pi pi pi ») et `message.wav`
            (message vocal en FRANÇAIS).
ENTRÉE    : un chemin de fichier (et le texte du message pour la voix).
SORTIE    : des fichiers WAV lus ensuite par le gestionnaire d'alerte.
PIPELINE  : préparation du bloc 9 (alerte).

POURQUOI  : générer les sons une fois pour toutes évite de dépendre, à chaque alerte, de la
            voix du système ou de la connexion internet. L'exécution reste 100 % hors ligne.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


class GenerateurSonnerie:
    """Crée une sonnerie faite de bips courts répétés : « pi pi pi pi pi pi pi pi ».

    Chaque bip est une sinusoïde pure (1 800 Hz) dont l'amplitude monte et descend très vite
    (enveloppe trapézoïdale) pour éviter les « claquements » au début et à la fin du bip :

        s(t) = enveloppe(t) × sin(2π f t)          avec f = 1 800 Hz

    Les bips sont séparés par de courts silences. Durée totale : environ 2 secondes.
    """

    FREQUENCE_ECHANTILLONNAGE = 44100   # échantillons par seconde
    FREQUENCE_BIP = 1800.0              # hertz : un « pi » aigu, bien audible dans un habitacle
    NOMBRE_BIPS = 8
    DUREE_BIP = 0.14                    # secondes
    PAUSE_ENTRE_BIPS = 0.10             # secondes de silence entre deux bips
    DUREE_MONTEE = 0.008                # secondes pour atteindre le volume plein
    DUREE_DESCENTE = 0.020              # secondes pour s'éteindre

    def generer(self, chemin: Path) -> None:
        """Écrit le fichier WAV de la sonnerie à l'emplacement indiqué.

        POURQUOI : aucun fichier audio externe n'est à chercher ou à télécharger.
        """
        donnees = self._synthetiser()
        Path(chemin).parent.mkdir(parents=True, exist_ok=True)
        # APPORT : le module standard `wave` écrit un fichier WAV sans dépendance externe.
        with wave.open(str(chemin), "wb") as fichier:
            fichier.setnchannels(1)                              # mono
            fichier.setsampwidth(2)                              # 2 octets = entiers 16 bits
            fichier.setframerate(self.FREQUENCE_ECHANTILLONNAGE)
            fichier.writeframes(donnees.tobytes())

    def _synthetiser(self) -> np.ndarray:
        """Calcule les échantillons du son (entiers 16 bits) : un bip, répété avec des silences.

        POURQUOI : une suite de bips brefs attire davantage l'attention qu'un son continu
        et se distingue des bruits de la route.
        """
        fe = self.FREQUENCE_ECHANTILLONNAGE
        nombre_echantillons_bip = int(fe * self.DUREE_BIP)
        # APPORT : `np.arange(n) / fe` construit l'axe du temps t (en secondes), un point par échantillon.
        temps = np.arange(nombre_echantillons_bip) / fe

        # Enveloppe trapézoïdale : montée rapide, plateau, descente rapide (valeurs de 0 à 1).
        montee = temps / self.DUREE_MONTEE
        descente = (self.DUREE_BIP - temps) / self.DUREE_DESCENTE
        enveloppe = np.clip(np.minimum(montee, descente), 0.0, 1.0)

        bip = enveloppe * np.sin(2.0 * np.pi * self.FREQUENCE_BIP * temps)
        silence = np.zeros(int(fe * self.PAUSE_ENTRE_BIPS))

        # APPORT : `np.concatenate` colle les bips et les silences bout à bout ; le dernier
        # silence est retiré pour que la pause d'une seconde avant la voix soit exacte.
        morceaux = []
        for numero in range(self.NOMBRE_BIPS):
            morceaux.append(bip)
            if numero < self.NOMBRE_BIPS - 1:
                morceaux.append(silence)
        signal = np.concatenate(morceaux)
        return (0.8 * 32767 * signal).astype(np.int16)   # 80 % du volume maximal


class GenerateurVoix:
    """Crée le message vocal en français.

    Deux moteurs sont essayés dans l'ordre :
      1. `pyttsx3` (hors ligne), uniquement si le système possède une VOIX FRANÇAISE ;
      2. sinon `gTTS` (voix de Google, nécessite internet UNE SEULE FOIS, à la génération).
    Si aucun des deux ne convient, une erreur claire explique quoi faire : on ne bascule
    JAMAIS en silence sur une voix anglaise.
    """

    def generer(self, chemin: Path, message: str) -> None:
        """Synthétise le message en français et l'enregistre dans un fichier WAV.

        POURQUOI : la génération n'a lieu qu'une fois ; ensuite le message se joue comme un
        son ordinaire, sans dépendre du moteur vocal.
        """
        Path(chemin).parent.mkdir(parents=True, exist_ok=True)
        if self._generer_avec_pyttsx3(Path(chemin), message):
            print("Voix française générée avec pyttsx3 (hors ligne).")
            return
        print("Pas de voix française utilisable avec pyttsx3 : essai avec gTTS (internet requis)...")
        if self._generer_avec_gtts(Path(chemin), message):
            print("Voix française générée avec gTTS.")
            return
        raise RuntimeError(
            "Impossible de générer le message vocal en français. Deux solutions :\n"
            "  1. installer une voix française sur votre système (Windows : Paramètres > Heure et "
            "langue > Voix > ajouter des voix > Français), puis relancer ce script ;\n"
            "  2. installer gTTS (pip install gTTS) et relancer ce script avec internet ;\n"
            "  ou enregistrer vous-même le message dans le fichier assets/message.wav."
        )

    def _choisir_voix_francaise(self, moteur) -> bool:
        """Sélectionne une voix française si le système en possède une ; renvoie le succès.

        POURQUOI : un message français lu avec une voix anglaise serait incompréhensible.
        Si aucune voix française n'existe, la liste des voix installées est affichée pour aider.
        """
        voix_installees = list(moteur.getProperty("voices"))
        for voix in voix_installees:
            langues = " ".join(
                langue.decode("utf-8", "ignore") if isinstance(langue, bytes) else str(langue)
                for langue in (getattr(voix, "languages", None) or [])
            )
            description = f"{voix.id} {voix.name} {langues}".lower()
            if any(marque in description for marque in ("fr-fr", "fr_fr", "french", "franç", "fran", "hortense", "julie", "paul", "/fr")):
                moteur.setProperty("voice", voix.id)
                print(f"Voix choisie : {voix.name}")
                return True
        print("Voix installées sur ce système :", [voix.name for voix in voix_installees])
        return False

    def _generer_avec_pyttsx3(self, chemin: Path, message: str) -> bool:
        """Essaie la synthèse hors ligne avec `pyttsx3` ; renvoie `True` si le fichier est créé.

        POURQUOI : c'est la solution la plus simple quand une voix française est installée.
        """
        try:
            # APPORT : `pyttsx3` utilise le moteur de synthèse du système (SAPI5 sous Windows,
            # espeak sous Linux) : il fonctionne sans connexion internet.
            import pyttsx3

            moteur = pyttsx3.init()
            if not self._choisir_voix_francaise(moteur):
                return False
            moteur.setProperty("rate", 150)   # débit de parole, en mots par minute
            moteur.save_to_file(message, str(chemin))
            moteur.runAndWait()
            return chemin.exists() and chemin.stat().st_size > 0
        except Exception as erreur:   # moteur absent, voix inutilisable…
            print(f"pyttsx3 indisponible : {erreur}")
            return False

    def _generer_avec_gtts(self, chemin: Path, message: str) -> bool:
        """Essaie la voix française de Google (`gTTS`), puis convertit le MP3 en WAV.

        POURQUOI : plan B quand le système n'a pas de voix française. Internet n'est nécessaire
        que pour cette génération ; ensuite le fichier est local.
        """
        try:
            # APPORT : `gTTS` envoie le texte au service de synthèse de Google et reçoit un MP3.
            from gtts import gTTS
        except ImportError:
            print("gTTS n'est pas installé (commande : pip install gTTS).")
            return False
        fichier_mp3 = chemin.with_suffix(".mp3")
        try:
            gTTS(text=message, lang="fr").save(str(fichier_mp3))
        except Exception as erreur:
            print(f"gTTS a échoué (internet disponible ?) : {erreur}")
            return False

        # Conversion MP3 -> WAV avec pygame, pour que le gestionnaire d'alerte lise un WAV.
        try:
            import pygame

            pygame.mixer.init()
            son = pygame.mixer.Sound(str(fichier_mp3))
            frequence, format_audio, canaux = pygame.mixer.get_init()
            with wave.open(str(chemin), "wb") as fichier:
                fichier.setnchannels(canaux)
                fichier.setsampwidth(abs(format_audio) // 8)
                fichier.setframerate(frequence)
                fichier.writeframes(son.get_raw())
            pygame.mixer.quit()
            return True
        except Exception as erreur:
            print(f"Conversion du MP3 impossible avec pygame : {erreur}")
            return False
