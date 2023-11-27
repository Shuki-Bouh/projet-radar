# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 08:24:30 2023

@author: amte7
"""
import numpy as np
import parametre as prm

class Target:
    nm = 0
    def __init__(self, r, theta, phi, vr):
        self.r = r
        self.theta = theta
        self.phi = phi
        self.vr = vr
        self.sequence = None
        self.num = Target.nm
        Target.nm += 1
        
    def calcul_difference_marche(self, channel):
        self.sequence = np.sum((channel - channel[:,0].reshape((2,1)))*np.array([np.sin(self.theta),-np.cos(self.theta)]).reshape(2,1), axis = 0)#+(np.abs(channel - channel[:,np.argmax(channel[1])].reshape((2,1))))[1]*np.sin(self.phi)
        #prend le reseau virtuelle, prend l'antenne la plus a gauche et calcul les differences de marche par rapport a elle                     #en prenant en compte l elevation
    

class Simulateur:
    def __init__(self):
        self.liste_activated_r = None
        self.liste_activated_t = None
        self.N_r = None
        self.N_t = None
        self.target = None
        self.R = None
        
        self.Nc = prm.Nc
        self.Tc = prm.Tc
        self.Ns = prm.Ns
        self.mode = [None]
        self.ddma = None
        self.tensor_ddma = None
        self.offset_ddma = None
    
        self.A_R =  np.array([[0, 1/2, 1, 3*1/2],
                            [0, 0, 0, 0]]) #antenne de reception

        self.A_T =  np.array([[0, 1, 2*1],
                            [0, 1/2, 0]]) #antenne de transmission 
        self.channel = None #coordone des antennes virtuelles
        self.Nch = None
        self.d = prm.λ/2
        self.S = prm.S
        self.c = prm.c
        self.fc = prm.fc
        self.lamb = prm.λ
        self.Time = None
        self.slow_time = None
        self.res_temp = None
        self.res = None
        
        
    def calcul_channel(self):
        selc = np.zeros((1, 4), dtype=bool)
        for k in self.liste_activated_r:
            selc[0, k-1] = True
        A = self.A_R[:, selc[0]] #garde que les coordondé des antennes  de reception activé
        
        bary = np.mean(A, axis=1).reshape((2,1)) #calcul du barycentre du reseau d'antenne rééles actives
        
        A = A - bary  
        for i in range(self.N_t):
            self.channel[:, i*self.N_r:(i+1)*self.N_r] = A + self.A_T[:, self.liste_activated_t[i]-1].reshape((2,1))  #la coordone de l'antenne d'emssion est le barycentre du reseau d'antenne rééles actives
            
    
    def calcul_temps(self):
        low_time = np.linspace(0,self.Nc*self.Tc,self.Nc, endpoint=False)  #le temps selon l echelle des chrips
        fast_time = np.linspace(0,self.Tc,self.Ns) #le temps selon l echelle de echantillonage
        lt, ft = np.meshgrid(low_time, fast_time)
        T = lt+ft #grille du temps 
        self.Time = np.repeat(T[:, :, np.newaxis], self.Nch, axis=2) #tenseur de la grille du temps, 1 grille par antenne virtuelle
        self.slow_time  =  np.repeat(lt[:, :, np.newaxis], self.Nch, axis=2) #tenseur de la grille du du temps cours (on considere la vitesse comme constante pendant un chrip et donc le temps pour la vitesse varie de chrip a chrip)
        
    def calcul_offset(self):
        self.tensor_ddma = np.zeros((self.Ns,self.Nc,self.N_r*self.N_t))
        i = 0
        for pas in self.offset_ddma:
            vecteur = np.linspace(0, 0 + (pas * (self.Nc - 1)), self.Nc) % (2*np.pi) #liste des dephasage a chaque chirp pour une certaine antenne d'emmision
            mat = np.meshgrid(vecteur, np.zeros((self.Ns)))[0]  #sous forme de grille 
            self.tensor_ddma[:,:,self.N_r*i:self.N_r*(i+1)] = np.repeat(mat[:, :, np.newaxis],self.N_r , axis=2) #sous frome de tenseur (1 grille par antenne virtuelle), les couche sous la forme TXi1,Rxj1|TXi1,Rxj2|TXi1,Rxj3|TXi2,Rxj1|TXi2,Rxj2|TXi2,Rxj3...
            i = i+1
        
    def calcul_sample(self):
        for tg in self.target:
            print(tg.num)
            tg.calcul_difference_marche(self.channel)
            A = np.broadcast_to(tg.sequence, (self.Ns, self.Nc, self.N_r*self.N_t))
            if self.ddma == 1:
                self.R = tg.r + tg.vr*self.slow_time +0.5*self.d*A + 0.5*self.ddma*self.tensor_ddma*self.lamb/(2*np.pi)
            else:
                self.R = tg.r + tg.vr*self.slow_time +0.5*self.d*A 
            self.res_temp = self.res_temp + np.exp(1j*(2*np.pi*2*self.S/self.c*self.Time*self.R+4*np.pi*self.R/self.lamb))
            
            
    def res_tdma(self):
#        for j in range(self.Nc):
#                self.res[:,j,:] = self.res_temp[:,j,(j%self.N_t)*self.N_r:((j%self.N_t)+1)*self.N_r]
        motif = np.tile(np.eye(self.N_t)[::-1],self.Nc//self.N_t)
        mask_2d = np.kron(motif,np.ones((self.N_r,1)))
        mask_3d = np.repeat(mask_2d[:,:,None], self.Ns ,axis = 2)
        mask_3d = np.transpose(mask_3d)
        mask_3d = np.fliplr(mask_3d)
        self.res_temp = mask_3d * self.res_temp
        
#        ll = np.fliplr(l)
#        l = np.transpose(nnn)
#        nnn = np.repeat(nn[:,:,None], 11 ,axis = 2)
#        nn = np.kron(matrice,np.ones((Nr,1)))
#        matrice = np.tile(matrice,Nc//Nt)
#        np.eye(Nt)[::-1]
            
            
    def channels_2_antennas(self):
#        indices = np.arange(self.N_r*self.N_t)
#        new_indices = np.array([])
#        for k in range(self.N_r):
#            new_indices =  np.concatenate([new_indices,indices[k::self.N_r]])
##            new_indices = np.concatenate([indices[0::self.N_r], indices[1::self.N_r],indices[2::self.N_r], indices[3::self.N_r]])
#        self.res_temp = self.res_temp[:,:,new_indices]
#        
        for k in range(self.N_r):
            self.res[:,:,k] = np.sum(self.res_temp[:,:,k::self.N_t], axis = 2)
        
    def run(self, liste_activatd_r,liste_activatd_t, mode,liste_targt):
        self.liste_activated_r = sorted(liste_activatd_r)
        self.liste_activated_t = sorted(liste_activatd_t)
        self.N_r = len(self.liste_activated_r)
        self.N_t = len(self.liste_activated_t) 
        self.ddma = 0
        self.dtma = 0
        self.tensor_ddma = None
        self.offset_ddma = np.zeros((self.N_t))
        self.target = liste_targt
        self.mode = mode
        if self.mode[0] == 'DTMA':
            self.dtma = 1
            self.Nc = self.Nc * self.N_t
        if self.mode[0] == 'DDMA':
            self.ddma = 1
            self.offset_ddma = self.mode[1]
        self.channel = np.zeros((2, len(self.liste_activated_r)*len(self.liste_activated_t))) #coordone des antennes virtuelles
        self.Nch = self.channel.shape[1]
        self.res_temp = np.zeros((self.Ns,self.Nc,self.N_r*self.N_t) , dtype=complex)
        self.res = np.zeros((self.Ns,self.Nc,self.N_r), dtype=complex)
        
        self.calcul_channel()
        self.calcul_temps()
        
        if self.ddma == 1:
            self.calcul_offset()
            
        self.calcul_sample()
        
        if self.dtma == 1:
            self.res_tdma()
            
        self.channels_2_antennas()
        
        return self.res
        
            
        
    