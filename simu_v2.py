# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 08:24:30 2023

@author: amte7
"""

import numpy as np
import parametre as prm

class Target:
    """
    Initialise un objet Target.

    Paramètres :
    - r : Distance de la cible par rapport à l'origine.
    - theta : Angle azimutal de la cible.
    - phi : Angle d'élévation de la cible.
    - vr : Vitesse radiale de la cible.
    - Amp : Amplitude du signal de la cible (par défaut 1).
    """
    nm = 0
    def __init__(self, r, theta, phi, vr,Amp=1):
        self.r = r
        self.theta = theta
        self.phi = phi
        self.vr = vr
        self.diff_marche = None
        self.num = Target.nm
        self.Amp = Amp
        Target.nm += 1
        
    def calcul_difference_marche(self, channel):
        """
        Calcule les différences de marche pour chaque antenne de réception par rapport à la cible.

        Paramètres :
        - channel : Coordonnées des antennes (virtuelles).

        Résultat :
        Les différences de marche sont stockées dans l'attribut diff_marche de l'objet Target.
        """

        self.diff_marche = np.sum((channel - channel[:,0].reshape((2,1)))*np.array([np.sin(self.theta),-np.cos(self.theta)]).reshape(2,1), axis = 0)
        #+(np.abs(channel - channel[:,np.argmax(channel[1])].reshape((2,1))))[1]*np.sin(self.phi) En prenant en compte l elevation, a voir dans le cas 3d si pas un pythagore qui traine.
        #prend le reseau virtuelle, prend l antenne la plus a gauche et calcul les differences de marche par rapport a elle.                  


class Simulateur:
    def __init__(self):
        """
        Initialise un objet Simulateur avec les paramètres par défaut.
        """
        
        self.liste_activated_r = None
        self.liste_activated_t = None
        self.N_r = None
        self.N_t = None
        self.target = None
        
        self.Nc = prm.Nc
        self.Tc = prm.Tc
        self.Ns = prm.Ns
        self.mode = [None]
        self.ddma = None
        self.tensor_ddma = None
        self.offset_ddma = None
    
        self.A_R =  np.array([[0, 1/2, 1, 3*1/2],
                            [0, 0, 0, 0]]) #antennes de reception

        self.A_T =  np.array([[0, 1, 2*1],
                            [0, 1/2, 0]]) #antennes de transmission 
        self.channel = None #coordonnees des antennes virtuelles
        self.Nch = None
        self.d = prm.λ/2
        self.S = prm.S
        self.c = prm.c
        self.fc = prm.fc
        self.lamb = prm.λ
        #self.Time = None
        self.slow_time = None
        self.fast_time = None
        self.result_channel = None
        self.result_true_antenna = None
        
        
    def calcul_channel(self):
        """
        Calcule les coordones du reseau d antennes de reception virtuelle
        """
        selc = np.zeros((1, 4), dtype=bool)
        for k in self.liste_activated_r:
            selc[0, k-1] = True
        A_R_activated = self.A_R[:, selc[0]] #les coordonees des antennes de reception actives.
        
        bary = np.mean(A_R_activated, axis=1).reshape((2,1)) #calcul du barycentre du reseau d antenne reeles actives.
        
        A_R_activated = A_R_activated - bary  
        for i in range(self.N_t):
            self.channel[:, i*self.N_r:(i+1)*self.N_r] = A_R_activated + self.A_T[:, self.liste_activated_t[i]-1].reshape((2,1))  #les antennes d'émission sont les barycentre de chaque reseau d antennes reelles actives.
    
    
    def calcul_temps(self):
        """
        Calcule les échelles de temps pour le simulateur.
        """
        
        low_time = np.linspace(0,self.Nc*self.Tc,self.Nc, endpoint=False)  #le temps selon l echelle des chrips
        fast_time = np.linspace(0,self.Tc,self.Ns) #le temps selon l echelle de l echantillonnage
        if self.ddma == 1:
            self.slow_time, fast_time,_ = np.meshgrid(low_time, fast_time, np.ones(self.Nch))
        else:
            self.slow_time, self.fast_time,_ = np.meshgrid(low_time, fast_time, np.ones(self.N_r))
        #self.Time = self.slow_time+self.fast_time #grille du temps 

    def calcul_offset(self):
        """
        Calcule les déphasages induits pour chaque chirp dans le cas d'une utilisation en DDMA.
        """        
        D = np.linspace(0, 0 + (self.offset_ddma * (self.Nc - 1)), self.Nc) %(2*np.pi)
        L = D.reshape(self.Nc,1,self.N_t)
        self.tensor_ddma = np.repeat(np.transpose(np.tile(L, (1, self.Ns, 1)), axes = (1,0,2)),repeats=self.N_r, axis=2)
 
    def calcul_sample(self):
        """
        Calcule les échantillons du signal IF en fonction des paramètres du simulateur et des cibles.
        """
        for tg in self.target:
            tg.calcul_difference_marche(self.channel)


            if self.tdma == 1:
                reorga_1 = np.arange(self.N_r*self.N_t).reshape((self.N_t, self.N_r)).T.flatten()
                p = tg.diff_marche[reorga_1].reshape((self.N_r,self.N_t))
                
                p = p.T.reshape((1,self.N_t,self.N_r))
                
                o = p + np.zeros((self.Ns,self.N_t,self.N_r))
                
                diff_marche_tensor = np.repeat(o, self.Nc//self.N_t, axis = 1)
                reorga_2 = np.arange(self.Nc).reshape((self.N_t, self.Nc//self.N_t)).T.flatten()
                diff_marche_tensor = diff_marche_tensor[:,reorga_2,:]
                
            else:    
                diff_marche_tensor = np.broadcast_to(tg.diff_marche, (self.Ns, self.Nc, self.N_r * self.N_t))
            


            if self.ddma == 1:
                R = tg.r + tg.vr * self.slow_time + 0.5 * self.d * diff_marche_tensor + 0.5 * self.ddma * self.tensor_ddma * self.lamb / (2 * np.pi)
            else:
                R = tg.r + tg.vr * self.slow_time + 0.5 * self.d * diff_marche_tensor

            self.result_channel +=tg.Amp * np.exp(
                1j * ((2 * np.pi * 2 * self.S / self.c * self.fast_time + 4 * np.pi / self.lamb) * R))
    def res_tdma(self):
        """
        Résultats pour le mode TDMA.
        """
        self.result_true_antenna = self.result_channel


            
            
    def channels_2_antennas(self):
        """
        Passe des canaux aux antennes réelles en sommant les canaux partageant la même antenne de réception.
        """        
        indices = np.arange(self.N_r*self.N_t).reshape((self.N_t, self.N_r)).T.flatten()
        self.result_channel = self.result_channel[:,:,indices].reshape((self.Ns,self.Nc,-1,self.N_t))
        self.result_true_antenna = np.sum(self.result_channel, axis = 3)
        

    def run(self, liste_activatd_r,liste_activatd_t, mode,liste_targt):
        """
        Exécute la simulation en fonction des paramètres spécifiés.

        Paramètres :
        - liste_activatd_r : Liste des antennes de réception activées.
        - liste_activatd_t : Liste des antennes de transmission activées.
        - mode : Mode de simulation ('TDMA', 'DDMA', 'SIMO', 'SISO')
        - liste_targt : Liste des cibles.

        Résultat :
        Retourne les résultats de la simulation pour les antennes réelles.
        """
        #gestion des erreurs
        if not isinstance(liste_activatd_r, list) or not isinstance(liste_activatd_t, list) or not  isinstance(mode, list) or not  isinstance(liste_targt, list):
            raise ValueError("at least one parameter is not a list")
        if not all(isinstance(tg, Target) for tg in liste_targt):
            raise ValueError("invalid target")
        if mode[0] == 'TDMA':
            if len(mode) != 1 or len(liste_activatd_t) <= 1:
                raise ValueError("Invalid parameters for TDMA mode.")
        elif mode[0] == 'DDMA':
            if len(mode) != 2 or len(liste_activatd_t) <= 1:
                raise ValueError("Invalid parameters for DDMA mode.")
            if type(mode[1]) != np.ndarray or mode[1].shape != (len(liste_activatd_t),):
                raise ValueError("Invalid offset for DDMA mode.")
        elif mode[0] == 'SIMO':
            if len(liste_activatd_t) != 1 or len(liste_activatd_r) <= 1:
                raise ValueError("pb in antenna count")
        elif mode[0] == 'SISO':
            if len(liste_activatd_t) != 1 or len(liste_activatd_r) != 1:
                raise ValueError("pb in antenna count")
        else:
            raise ValueError("invalid mode")
                
        if not (self.liste_activated_r == liste_activatd_r and self.liste_activated_t == liste_activatd_t):
            if  not all(isinstance(i, int) for i in liste_activatd_r) or len(set(liste_activatd_r)) != len(liste_activatd_r)or max(liste_activatd_r) >4 or min(liste_activatd_r)<1:
                raise ValueError("Invalid parameter for liste_activated_r. It should be a list of unique integers .")

            if not isinstance(liste_activatd_t, list) or not all(isinstance(i, int) for i in liste_activatd_t) or len(set(liste_activatd_t)) != len(liste_activatd_t) or max(liste_activatd_t) >3 or min(liste_activatd_t)<1:
                raise ValueError("Invalid parameter for liste_activated_t. It should be a list of unique integers.")
                
            self.liste_activated_r = sorted(liste_activatd_r)
            self.liste_activated_t = sorted(liste_activatd_t)
            self.N_r = len(self.liste_activated_r)
            self.N_t = len(self.liste_activated_t) 
            self.channel = np.zeros((2, len(self.liste_activated_r)*len(self.liste_activated_t))) #coordone des antennes virtuelles
            self.Nch = self.channel.shape[1]
            self.calcul_channel()
        self.Nc = prm.Nc
        self.ddma = 0
        self.tdma = 0
        self.tensor_ddma = None
        self.offset_ddma = np.zeros((self.N_t))
        self.target = liste_targt
        self.mode = mode
        if self.mode[0] == 'TDMA':
            self.tdma = 1
            self.Nc = self.Nc * self.N_t
        if self.mode[0] == 'DDMA':
            self.ddma = 1
            self.offset_ddma = self.mode[1]
        
        if self.tdma !=1:
            self.result_channel = np.zeros((self.Ns,self.Nc,self.N_r*self.N_t) , dtype=complex)
        else:
            self.result_channel = np.zeros((self.Ns,self.Nc,self.N_r) , dtype=complex)
        self.result_true_antenna = np.zeros((self.Ns,self.Nc,self.N_r), dtype=complex)
        
        self.calcul_temps()
        
        if self.ddma == 1:
            self.calcul_offset()
            
        self.calcul_sample()
        
        if self.tdma == 1:
            self.res_tdma()
        else:
            self.channels_2_antennas()
        
        return self.result_true_antenna
        
            
        
    
