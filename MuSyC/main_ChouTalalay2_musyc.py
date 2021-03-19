from synergy.combination import MuSyC # or BRAID, Zimmer
from synergy.combination import Loewe, Bliss # or Bliss, ZIP, HSA, Schindler, CombinationIndex
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from synergy.utils import plots

from synergy.utils.dose_tools import grid
import numpy as np

np.random.seed(100)

df = pd.read_csv("../data/Yonetani2_weighted.csv")

df['Effect'] = df['Effect']/1000.0

A_max = np.max(df['Dose_A'].to_numpy())
B_max = np.max(df['Dose_B'].to_numpy())

df_original = pd.read_csv("../data/YonetaniData2.csv", sep = ";")

df_original['FracInhib'] = df_original['FracInhib']/1000.0

Effect_AB = df_original['FracInhib'].values.reshape(-1,1)
Effect_B = df_original[df_original['Conc.ADP'] == 0]['FracInhib'].to_numpy().reshape(-1,1).astype(float)
Effect_A = df_original[df_original['Conc.Phen'] == 0]['FracInhib'].to_numpy().reshape(-1,1).astype(float)
Dose_AA = df_original[df_original['Conc.Phen']==0]['Conc.ADP'].to_numpy().reshape(-1,1).astype(float)
Dose_BB = df_original[df_original['Conc.ADP']==0]['Conc.Phen'].to_numpy().reshape(-1,1).astype(float)

Dose_A = df_original['Conc.ADP'].to_numpy().reshape(-1,1).astype(float)
Dose_B = df_original['Conc.Phen'].to_numpy().reshape(-1,1).astype(float)


# create df for diagonal

data = {'drug1.conc':  [0.5, 1.0, 1.5, 2.0, 2.5],
        'drug2.conc': [8.7, 17.4, 26.1, 34.8, 43.5],
         'effect': [493.0/1000.0, 231.0/1000.0, 128.0/1000.0, 81.0/1000.0, 56.0/1000.0]
        }

df_diagonal = pd.DataFrame (data, columns = ['drug1.conc','drug2.conc','effect'])

# One time point

df = pd.concat([df['Dose_A'], df['Dose_B'], df['Effect']], axis=1)
df.columns=['drug1.conc', 'drug2.conc', 'effect']

plt.figure(figsize=(8, 8))
plt.scatter(np.asarray(df['drug1.conc']), np.asarray(df['drug2.conc']))
#plt.plot(np.asarray(df['drug1.conc']), np.asarray(df['drug2.conc']),'o'), plt.axis("normal")
# Annotate plot
plt.xlabel("$x_1$"), plt.ylabel("$x_2$")
plt.savefig("figures/ChouTalalay2/ChouTalalay2_data_points.png")

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
df_result.to_csv('results/ChouTalalay2/df_result_ChouTalalay2.csv')

# plot
plt.figure(figsize=(12, 6))
plt.plot(dose_a.flatten(), effect_a.flatten(), "kx", mew=2)
plt.plot(np.linspace(0.1, A_max, num=100), Effect_musyc[0,:], "purple", lw=2)
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('Response', fontsize=20)
plt.ylim(0.0,1.1)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('figures/ChouTalalay2/ChouTalalay_musyc2'+'drugA'+'.png')

# plot
plt.figure(figsize=(12, 6))
plt.plot(dose_b.flatten(), effect_b.flatten(), "kx", mew=2)
plt.plot(np.linspace(0.1, B_max, num=100), Effect_musyc[:,0], "purple", lw=2)
plt.xlabel('$x_2$', fontsize=20)
plt.ylabel('Response', fontsize=20)
plt.ylim(0.0,1.1)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('figures/ChouTalalay2/ChouTalalay_musyc2'+'drugB'+'.png')

fig, ax = plt.subplots(figsize=(6,6))
v = np.linspace(-0.05, 0.05, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(Dose_AA.flatten(), Dose_BB.flatten(), (model.E(Dose_A, Dose_B) - Effect_AB).flatten().reshape(6,6),v, cmap='YlGnBu')
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.savefig('figures/ChouTalalay2/Difference_musyc_ChouTalalay2.png', bbox_inches = 'tight',
    pad_inches = 0)

fig, ax = plt.subplots(figsize=(6,6))
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
v = np.linspace(0.0, 1.05, 10, endpoint=True)
cf = ax.contourf(Dose_AA.flatten(), Dose_BB.flatten(), model.E(Dose_A, Dose_B).flatten().reshape(6,6),v)
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
    t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.savefig('figures/ChouTalalay2/Surface_musyc_ChouTalalay2.png', bbox_inches = 'tight',
    pad_inches = 0)

# Create doses on the diagonal and predict in the diagonal values

# Thw equation for the diagonal: y = 17.4*x
x = np.linspace(0.0,2.5,100)
y = 17.4 * x

#data = {'drug1.conc':  [0.5, 1.0, 1.5, 2.0, 2.5],
#        'drug2.conc': [8.7, 17.4, 26.1, 34.8, 43.5],
#         'effect': [493.0, 231.0, 128.0, 81.0, 56.0]
#        }
#df_diagonal = pd.DataFrame (data, columns = ['drug1.conc','drug2.conc','effect'])

Effect_musyc_diagonal = model.E(x, y)

plt.figure(figsize=(12, 6))
plt.plot(np.asarray(np.sqrt(x**2+y**2)).flatten(), Effect_musyc_diagonal.flatten(), "r", mew=2)
plt.plot(np.sqrt(df_diagonal['drug1.conc']**2+df_diagonal['drug2.conc']**2), df_diagonal['effect'], "kx", lw=2)
plt.xlabel('$x_1$+$x_2$', fontsize=20)
plt.ylabel('Response', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('figures/ChouTalalay2/ChouTalalay_musyc2'+'_diagonal'+'.png')

# Null model MuSyC
model = MuSyC(variant='no_gamma',alpha12_bounds=(1,1),alpha21_bounds=(1,1))
model.fit(df['drug1.conc'], df['drug2.conc'], df['effect'],bootstrap_iterations=100)
ci_no_gamma = model.get_parameters()

fig, ax = plt.subplots(figsize=(6,6))
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
v = np.linspace(0.0, 1.05, 10, endpoint=True)
cf = ax.contourf(Dose_AA.flatten(), Dose_BB.flatten(), (model.E(Dose_A, Dose_B)).flatten().reshape(6,6),v)
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.savefig('figures/ChouTalalay2/Surface_null_musyc_ChouTalalay2.png', bbox_inches = 'tight',
    pad_inches = 0)

fig, ax = plt.subplots(figsize=(6,6))
v = np.linspace(-0.06, 0.42, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(Dose_AA.flatten(), Dose_BB.flatten(), (model.E(Dose_A, Dose_B) - Effect_AB).flatten().reshape(6,6),v, cmap='RdYlGn')
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.savefig('figures/ChouTalalay2/Difference_musyc_ChouTalalay2_2.png', bbox_inches = 'tight',
    pad_inches = 0)

Effect_musyc_diagonal_null = model.E(x, y)

plt.figure(figsize=(12, 6))
plt.plot(np.asarray(np.sqrt(x**2+y**2)).flatten(), Effect_musyc_diagonal_null.flatten(), "grey", mew=2, label='Null')
plt.plot(np.asarray(np.sqrt(x**2+y**2)).flatten(), Effect_musyc_diagonal.flatten(), "purple", mew=2, label='Unconstrained')
plt.plot(np.sqrt(df_diagonal['drug1.conc']**2+df_diagonal['drug2.conc']**2), df_diagonal['effect'], "kx", lw=2)
plt.xlabel('Diagonal combination of the doses', fontsize=20)
plt.ylabel('Response', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend( prop={'size': 20})
plt.savefig('figures/ChouTalalay2/ChouTalalay2_musyc'+'_diagonal_null'+'.png', bbox_inches = 'tight',
    pad_inches = 0)

xnew = np.asarray(np.sqrt(x**2+y**2)).flatten()
xnew = (xnew - xnew.min())/(xnew.max()-xnew.min())

plt.figure(figsize=(12, 6))
plt.plot(Effect_musyc_diagonal[::-1][0:2], (Effect_musyc_diagonal_null.flatten()-Effect_musyc_diagonal.flatten())[0:2], "red", mew=2)
plt.plot(Effect_musyc_diagonal[::-1][1:len(np.asarray(np.sqrt(x**2+y**2)).flatten())], (Effect_musyc_diagonal_null.flatten()-Effect_musyc_diagonal.flatten())[1:len(np.asarray(np.sqrt(x**2+y**2)).flatten())], "green", mew=2)
plt.xlabel('Fractional effect', fontsize=20)
plt.ylabel('Difference', fontsize=20)
plt.ylim(-0.5,0.5)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('figures/ChouTalalay2/ChouTalalay2_musyc'+'_diagonal_null_difference'+'.png', bbox_inches = 'tight',
    pad_inches = 0)


print('Finished')
