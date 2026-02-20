#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from classy import Class
import time
from scipy.integrate import quad
from scipy.interpolate import interp1d
import sys
import corecon as con
print("\n" + "="*60)
print("RUNNING FIDUCIAL MODEL")
print("="*60)

#dando in ingresso un array di z restituisce xe(z)
#===========================================================
# Nota: Delta_z e' fisso a 0.5 perche' ho una griglia piu' fitta, quindi non rischio divergenze.
#  quindi in questo caso il sampler sara' su z_re
#===========================================================

def custom_xe_tanh(z, z_re=8.5, Delta_z=0.5):
    """
    Storia di reionizzazione con tanh standard
    Parameters:
    - z: array di redshift in ingresso 
    - z_re: redshift centrale della reionizzazione, questo lo vario
    - Delta_z: larghezza della transizione, ma questa la fisso 
    - 1.08 sarebbe la normalizzazione per includere l'effetto dell'elio.
    -xe(z) comunque descrive tutti gli elettroni liberi, che comprendono anche quelli dell'elio.
    """
    # Normalizzazione per avere xe ~ 1.08 dopo la reionizzazione (include He)
    xe = 1.08 * np.tanh((z_re-z) / Delta_z) / 2.0 + 1.08 / 2.0
    return xe

z_range=np.linspace(0, 30, 100)
xe_fiducial = custom_xe_tanh(z=z_range, z_re=8.5, Delta_z=0.5) #plotting purpose
#x_e per diversi z_re definisce storie diverse della reinionizzazione con stesso Delta_z.
#valore fiduciale per z_re=8.5 e Delta_z=0.5
x_e=custom_xe_tanh(z=z_range, z_re=8.5, Delta_z=0.5)
Y_He = 0.24  # primordial helium fraction
fHe = Y_He / (1 - Y_He) * (1.008 / 4.003)  # ≈ 0.079
x_H = x_e / (1 + fHe)          # ionizzazione idrogeno
x_H_neutral = 1 - x_H  # frazione neutra totale di idrogeno e elio

Dai_def=1-x_e/1.08 #frazione neutra di solo idrogeno, questo e' x_HI
# print("la differenza tra le due frazioni di idrogeno e': ", np.max(np.abs(Dai_def - x_H_neutral)))

plt.plot(z_range, x_H_neutral, label='Neutral Hydrogen Fraction (Fiducial Model)')
plt.plot(z_range,Dai_def, label='Neutral Hydrogen Fraction (Definition)', linestyle=':')
plt.plot(z_range, xe_fiducial, label='Free Electron Fraction (Fiducial Model)', linestyle='--')
plt.xlabel('Redshift z')
plt.ylabel('Neutral Hydrogen Fraction x_HI(z)')
plt.title('Fiducial Reionization Model: Neutral Hydrogen Fraction vs Redshift')
plt.ylim(0, 1.2)
plt.xlim(0, 30)
plt.grid()
plt.legend()
plt.show()
# exit()
#===========================================
# Estraggo constraints da corecon
#===========================================

xhii_corecon_full_vector= []
z_xhii_corecon_full_vector=[]

dict=con.get("x_HII")
xhii_corecon= dict["Umeda et al. 2023 (subm.)"] #cosi' recupero tutte le informazioni
print(dict["Umeda et al. 2023 (subm.)"])
xhii_corecon= dict["Umeda et al. 2023 (subm.)"].values 
xhii_corecon_full_vector.append(xhii_corecon)
z_xhii_corecon=dict["Umeda et al. 2023 (subm.)"].axes
z_xhii_corecon_full_vector.append(z_xhii_corecon)
print(z_xhii_corecon_full_vector)

dict=con.get("x_HII")
xhii_corecon= dict["Nakane et al. 2024"] #cosi' recupero tutte le informazioni
xhii_corecon= dict["Nakane et al. 2024"].values 
xhii_corecon_full_vector.append(xhii_corecon)
z_xhii_corecon=dict["Nakane et al. 2024"].axes
z_xhii_corecon_full_vector.append(z_xhii_corecon)
print(z_xhii_corecon_full_vector)
print(xhii_corecon_full_vector)