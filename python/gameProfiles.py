#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:46:41 2022

@author: rve
"""

import matplotlib.pyplot as plt
import numpy as np

# class Profile:
#     def __init__(self, acronym, name, production, consumption):
#         self.acronym = acronym
#         self.name = name
#         self.production = production
#         self.consumption = consumption


# Information extracted from the PHP project:
seasons = ["hiver", "printemps", "été", "automne"]
acronyms = ["P1", "P2", "P3", "P4", "P5"]
names = ["Entreprise sans panneau", "Maison sans panneau", "Église avec panneaux", "Maison avec panneaux", "Hôpital avec panneaux"]
productions = [
 [[0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0]],
 [[0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0]],
 [[4, 4, 4, 4, 4, 4, 4],
  [6, 6, 6, 6, 6, 6, 6],
  [9, 9, 9, 9, 9, 9, 9],
  [6, 6, 6, 6, 6, 6, 6]],
 [[2, 2, 2, 2, 2, 2, 2],
  [4, 4, 4, 4, 4, 4, 4],
  [6, 6, 6, 6, 6, 6, 6],
  [4, 4, 4, 4, 4, 4, 4]],
 [[8, 8, 8, 8, 8, 8, 8],
  [12, 12, 12, 12, 12, 12, 12],
  [15, 15, 15, 15, 15, 15, 15],
  [12, 12, 12, 12, 12, 12, 12]] ]
consumptions = [
  [[11, 11, 11, 11, 10, 0, 0],
   [9, 9, 9, 9, 9, 0, 0],
   [6, 6, 6, 6, 6, 0, 0],
   [10, 10, 10, 10, 9, 0, 0]],
  [[6, 6, 6, 6, 6, 9, 9],
   [3, 3, 3, 3, 3, 7, 7],
   [1, 1, 1, 1, 1, 4, 4],
   [4, 4, 4, 4, 4, 7, 7]],
  [[0, 0, 0, 0, 0, 4, 8],
   [0, 0, 0, 0, 0, 2, 6],
   [0, 0, 0, 0, 0, 1, 2],
   [0, 0, 0, 0, 0, 2, 6]],
  [[6, 6, 6, 6, 6, 8, 8],
   [4, 4, 4, 4, 4, 6, 6],
   [2, 2, 2, 2, 2, 4, 4],
   [4, 4, 4, 4, 4, 6, 6]],
  [[18, 18, 18, 18, 18, 18, 18],
   [16, 16, 16, 16, 16, 16, 16],
   [15, 15, 15, 15, 15, 15, 15],
   [16, 16, 16, 16, 16, 16, 16]] ]


# profiles = []
# for i in range(len(acronyms)):
#     profile = Profile(acronyms[i], names[i], productions[i], consumptions[i])
#     profiles.append(profile)


# Year of 336 days (12 weeks for each season)
prodsAnnuel = [] # Production of each profile during 1 year
conssAnnuel = [] # Consumption of each profile during 1 year
for i in range(len(acronyms)):
    phiv = productions[i][0]*12
    ppri = productions[i][1]*12
    pete = productions[i][2]*12
    paut = productions[i][3]*12
    pannuel = phiv + ppri + pete + paut
    prodsAnnuel.append(pannuel)
    chiv = consumptions[i][0]*12
    cpri = consumptions[i][1]*12
    cete = consumptions[i][2]*12
    caut = consumptions[i][3]*12
    cannuel = chiv + cpri + cete + caut
    conssAnnuel.append(cannuel)


plt.figure()
for i in range(len(acronyms)):
    plt.plot(range(len(prodsAnnuel[i])), prodsAnnuel[i], label=acronyms[i]) #, marker='o', linestyle='-', markersize=3
plt.title("Production")
plt.xlabel("Days")
plt.ylabel("kW")
plt.legend()
#plt.savefig("production.png")

plt.figure()
for i in range(len(acronyms)):
    plt.plot(range(len(conssAnnuel[i])), conssAnnuel[i], label=acronyms[i])
plt.title("Consumption")
plt.xlabel("Days")
plt.ylabel("kW")
plt.legend()


prodsAnnuel = np.array(prodsAnnuel)
conssAnnuel = np.array(conssAnnuel)
prodCom = prodsAnnuel.sum(axis=0)
consCom = conssAnnuel.sum(axis=0)
plt.figure()
plt.plot(range(len(prodCom)), prodCom, label='Production')
plt.plot(range(len(consCom)), consCom, label='Consumption')
plt.title("Community's Production / Consumption")
plt.xlabel("Days")
plt.ylabel("kW")
plt.legend()
