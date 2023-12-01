## Classe `Target`

La classe `Target` représente une cible radar.

- `r` (float): Distance (en mètres) de la cible par rapport à l'origine.
- `theta` (float): Angle azimutal de la cible en radians, entre -pi/2 et pi/2.
- `phi` (float): Angle d'élévation de la cible en radians.
- `vr` (float): Vitesse radiale de la cible en m/s.
- `Amp` (float): Amplitude du signal de la cible (par défaut 1).

**Exemple:**
```python
tg1 = Target(10, np.deg2rad(60), np.deg2rad(10), 2, 0.7)

tg2 = Target(20, np.deg2rad(-40), np.deg2rad(0), 1.4)
```
## Classe `Simulateur`

La classe `Simulateur` permet de simuler le radar avec différentes configurations.
```python
simu = Simulateur()
```
### Méthode `run`

```python
Simulateur.run(liste_activated_r, liste_activated_t, mode, liste_target)
```
renvoie le cube radar
- `liste_activatd_r` (liste de int): Liste des antennes de réception activées. 
- `liste_activatd_t` (liste de int): Liste des antennes de transmission activées. 
- `mode` (liste): 
                  <br/> `liste[0]` (str) : mode de simulation ('TDMA', 'DDMA', 'SIMO', 'SISO')
                    <br/> si `liste[0] == 'DDMA'`, `liste[1]` (np.array de shape (len(liste_activatd_t),) ) prenant les offsets de phase (en radian) pour chaque antenne d'emission
- `targt` : Liste des cibles. (liste de Target)

**Exemple:**
```python

cube = simu.run([1, 2, 3, 4], [1, 2, 3], ['TDMA'], [tg1, tg2])
# ou
cube = simu.run([1, 2, 3, 4], [1, 2, 3], ['DDMA', np.deg2rad(np.array([0, 90, 270]))], [tg1, tg2])
# ou
cube = simu.run([1, 2, 3, 4], [1], ['SIMO'], [tg1, tg2])
# ou
cube = simu.run([4], [3], ['SISO'], [tg1, tg2])
```
on peut aussi acceder au cube, une fois simu.run deja runné avec cube = simu.result_true_antenna                  
