fc = 77 * 10 ** 9  # fréquence basse

Nt = 1  #n b de Tx
Nr = 4  # nb de Rx
Nc = 128  # nb de chirp par frame
Nf = 200  # nb de frame
Ns = 256  # nb de sample par chirp
isReal = False

Tc = 60.17 * 10 ** -6 # durée d'un chirp
S = 15.015*10**6/10**-6  # slope
B = Tc*S  # Bandwidth

fs = Ns/Tc  # fréquence d'échantillonage
c = 3e8  # célérité
λ = c/fc  # longueur d'onde basse