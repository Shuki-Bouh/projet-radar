AWR1843BOOST
-

Note : J'utilise la version 3.2 du SDK TI pour l'AWR1843BOOST

**Prérequis :**
  - Driver TI : https://www.ti.com/tool/download/MMWAVE-SDK/03.02.01.02 disponible en exe et bin (prévu pour x86 et arm) C:\ti\mmwave_studio_02_01_01_00\ftdi
  - Il faut également utiliser les extentions et le TI cloud agent (tout se télécharge depuis le SDK)
  - Le navigateur à utiliser est Google Chrome 

**Connexion :**
AWR1843BOOST est alimentée par jack 5V 2.5A min et connecté au PC par un micro USB. Celui-ci sert à configurer le radar initialement
et à visualiser les données sur le SDK : https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.5.0/
Le premier port (config) se connecte tout seul lorsque le driver est correctement installé, le deuxième port (data) se connecte lorsqu'on 
envoie la configuration (configurer en xWR18xx v.3.2)

**Configuration :**
Il est possible de créer le fichier config.cfg qu'on télécharge depuis le SDK sans connecter la carte.
Le port de config a un nombre de baud de 115200 et le port data de 921600

DCA1000 
-
