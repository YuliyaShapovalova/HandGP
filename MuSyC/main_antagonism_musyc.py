from synergy.combination import MuSyC # or BRAID, Zimmer
from synergy.combination import Loewe, Bliss # or Bliss, ZIP, HSA, Schindler, CombinationIndex
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from synergy.utils import plots

from synergy.utils.dose_tools import grid
import numpy as np

np.random.seed(100)

df = pd.read_csv("../data/LA_antagonism.csv")

A_max = np.max(df['DRUG_A'].to_numpy())
B_max = np.max(df['DRUG_B'].to_numpy())

Effect_AB = df['E'].values.reshape(-1,1)
Effect_B = df[df['DRUG_A'] == 0]['E'].to_numpy().reshape(-1,1).astype(float)
Effect_A = df[df['DRUG_B'] == 0]['E'].to_numpy().reshape(-1,1).astype(float)
Dose_AA = df[df['DRUG_B']==0]['DRUG_A'].to_numpy().reshape(-1,1).astype(float)
Dose_BB = df[df['DRUG_A']==0]['DRUG_B'].to_numpy().reshape(-1,1).astype(float)

Dose_A = df['DRUG_A'].to_numpy().reshape(-1,1).astype(float)
Dose_B = df['DRUG_B'].to_numpy().reshape(-1,1).astype(float)

# One time point

df = pd.concat([df['DRUG_A'], df['DRUG_B'], df['E']], axis=1)
df.columns=['drug1.conc', 'drug2.conc', 'effect']

single_dose_a = df[df['drug2.conc']==0]
single_dose_b = df[df['drug1.conc']==0]

dose_a = single_dose_a['drug1.conc'].to_numpy()
effect_a = single_dose_a['effect'].to_numpy()

dose_b = single_dose_b['drug2.conc'].to_numpy()
effect_b = single_dose_b['effect'].to_numpy()

model = MuSyC()
model.fit(df['drug1.conc'], df['drug2.conc'], df['effect'],bootstrap_iterations=100)
model.get_parameters(confidence_interval=95)

#d11 = np.linspace(0.0, 14.0, num=100)
#d22 = np.linspace(0.0, 11.0, num=100)

[d11, d22] = np.meshgrid(np.linspace(0.1, A_max, num=100),  np.linspace(0.1, B_max, num=100))

Effect_musyc = model.E(d11, d22)
Effect_musyc = Effect_musyc.reshape((len(d11), len(d22)))

ci = model.get_parameters(confidence_interval=95)

df_ci = pd.DataFrame(np.concatenate([ci['beta'][1].reshape(1,2), ci['alpha12'][1].reshape(1,2), ci['alpha21'][1].reshape(1,2),
ci['gamma12'][1].reshape(1,2), ci['gamma21'][1].reshape(1,2)], axis=0)).round(2)
df_estimate = pd.DataFrame(np.concatenate([ci['beta'][0].reshape(1,1), ci['alpha12'][0].reshape(1,1), ci['alpha21'][0].reshape(1,1),
ci['gamma12'][0].reshape(1,1), ci['gamma21'][0].reshape(1,1)], axis=0)).round(2)

# Create data frame of the estimates of the parameters and confidence intervals to save in .csv file
df_result = pd.concat([df_estimate, df_ci], axis=1)
df_result.columns=['estimate', 'ci_l', 'ci_u']
df_result.index=['beta', 'alpha12', 'alpha21', 'gamma12', 'gamma21']
df_result.to_csv('results/LA_antagonism/df_result_LA_antagonism.csv')

'''
Plot monotherapeutic slices
'''
# plot
plt.figure(figsize=(12, 6))
plt.plot(dose_a.flatten(), effect_a.flatten(), "kx", mew=2)
plt.plot(np.linspace(0.1, A_max, num=100), Effect_musyc[0,:], "r", lw=2)
plt.savefig('figures/LA_antagonism/Loewesantagonism_musyc'+'drugA'+'.png')

# plot
plt.figure(figsize=(12, 6))
plt.plot(dose_b.flatten(), effect_b.flatten(), "kx", mew=2)
plt.plot(np.linspace(0.1, B_max, num=100), Effect_musyc[:,0], "r", lw=2)
plt.savefig('figures/LA_antagonism/Loeweantagonism_musyc'+'drugB'+'.png')

'''
Plot residuals
'''
fig, ax = plt.subplots(figsize=(6,6))
#ax.set_aspect('equal')
v = np.linspace(-0.1, 0.1, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(Dose_AA.flatten(), Dose_BB.flatten(), (model.E(Dose_A, Dose_B) - Effect_AB).flatten().reshape(11,11), v,  cmap='YlGnBu')
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
#plt.title("MuSyC residuals", fontsize=20)
plt.savefig('figures/LA_antagonism/Difference_musyc_Loeweantagonism.png',bbox_inches = 'tight',
    pad_inches = 0)

fig, ax = plt.subplots(figsize=(6,6))
v = np.linspace(-0.15, 1.05, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(Dose_AA.flatten(), Dose_BB.flatten(), model.E(Dose_A, Dose_B).flatten().reshape(11,11),v)
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
#plt.title("MuSyC estimated surface", fontsize=20)
plt.savefig('figures/LA_antagonism/Surface_musyc_Loeweantagonism.png',bbox_inches = 'tight',
    pad_inches = 0)

'''
Define and optimize constrained (null) model
'''
model = MuSyC(variant='no_gamma',alpha12_bounds=(1,1),alpha21_bounds=(1,1))
model.fit(df['drug1.conc'], df['drug2.conc'], df['effect'],bootstrap_iterations=100)
ci_no_gamma = model.get_parameters()

fig, ax = plt.subplots(figsize=(6,6))
v = np.linspace(-0.15, 1.05, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
v = np.linspace(-0.15, 1.05, 10, endpoint=True)
cf = ax.contourf(Dose_AA.flatten(), Dose_BB.flatten(), (model.E(Dose_A, Dose_B)).flatten().reshape(11,11),v)
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.savefig('figures/LA_antagonism/Surface_null_musyc_LA_antagonism.png',bbox_inches = 'tight',
    pad_inches = 0)

fig, ax = plt.subplots(figsize=(6,6))
v = np.linspace(-1, 1, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(Dose_AA.flatten(), Dose_BB.flatten(), (Effect_AB - model.E(Dose_A, Dose_B)).flatten().reshape(11,11),v, cmap='RdYlGn')
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.savefig('figures/LA_antagonism/Difference_musyc_LA_antagonism_2.png',bbox_inches = 'tight',
    pad_inches = 0)

print('Finished')
