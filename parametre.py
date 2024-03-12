import numpy as np

fc = 77 * 10 ** 9  # fréquence basse

Nt = 1  #nb de Tx
Nr = 4  # nb de Rx
Nc = 128  # nb de chirp par frame
Nf = 200  # nb de frame
Ns = 256  # nb de sample par chirp
isReal = False
mode = 'SIMO'

Tc = 60 * 10 ** -6 # durée d'un chirp (ramp end time)
S = 29.982*10**6/10**-6  # slope (Mhz/us)
B = Tc*S  # Bandwidth

Tprt = 100*10**-6 # durée entre deux chirps

fs = Ns/Tc  # fréquence d'échantillonage
c = 3e8  # célérité
λ = c/fc  # longueur d'onde basse

dmax = (fs*c)/(2*S)
dres = c/(2*B)

vmax = λ/(4*Tprt)
vres = λ/(2*Nc*Tprt)

thetares = np.rad2deg(2 / Nr)





