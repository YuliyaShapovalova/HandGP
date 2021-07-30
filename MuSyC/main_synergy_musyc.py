from synergy.combination import MuSyC # or BRAID, Zimmer
from synergy.combination import Loewe, Bliss # or Bliss, ZIP, HSA, Schindler, CombinationIndex
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from synergy.utils import plots
import scipy

from synergy.utils.dose_tools import grid
import numpy as np

def trapezoidal_area(xyz):
    """Calculate volume under a surface defined by irregularly spaced points
    using delaunay triangulation. "x,y,z" is a <numpoints x 3> shaped ndarray."""
    d = scipy.spatial.Delaunay(xyz[:,:2])
    tri = xyz[d.vertices]

    a = tri[:,0,:2] - tri[:,1,:2]
    b = tri[:,0,:2] - tri[:,2,:2]
    proj_area = np.cross(a, b).sum(axis=-1)
    zavg = tri[:,:,2].sum(axis=1)
    vol = zavg * np.abs(proj_area) / 6.0
    return vol.sum()

np.random.seed(100)

df = pd.read_csv("../data/LA_synergy.csv")

print(df)

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

'''
Define and optimize unconstrained model
'''

model = MuSyC(E1_bounds=(0.0,100.0),E2_bounds=(0.0,100.0), E3_bounds=(0.0,100.0))
model.fit(df['drug1.conc'], df['drug2.conc'], df['effect'],bootstrap_iterations=100)
model.get_parameters(confidence_interval=95)

ci = model.get_parameters(confidence_interval=95)

print(ci)

n_boot = 500
n_predict = len(model.E(df['drug1.conc'], df['drug2.conc']).to_numpy().flatten())

bootstrap_samples = np.zeros((n_boot, n_predict))

for i in range(n_boot):
    model.E0 = np.random.uniform(ci['E0'][1][0],ci['E0'][1][1],1)[0]
    model.E1 = np.random.uniform(ci['E1'][1][0],ci['E1'][1][1],1)[0]
    model.E2 = np.random.uniform(ci['E2'][1][0],ci['E2'][1][1],1)[0]
    model.E3 = np.random.uniform(ci['E3'][1][0],ci['E3'][1][1],1)[0]

    model.h1 = np.random.uniform(ci['h1'][1][0],ci['h1'][1][1],1)[0]
    model.h2 = np.random.uniform(ci['h2'][1][0],ci['h2'][1][1],1)[0]
    model.C1 = np.random.uniform(ci['C1'][1][0],ci['C1'][1][1],1)[0]
    model.C2 = np.random.uniform(ci['C2'][1][0],ci['C2'][1][1],1)[0]

    model.beta = np.random.uniform(ci['beta'][1][0],ci['beta'][1][1],1)[0]
    model.alpha12 = np.random.uniform(ci['alpha12'][1][0],ci['alpha12'][1][1],1)[0]
    model.alpha21 = np.random.uniform(ci['alpha21'][1][0],ci['alpha21'][1][1],1)[0]
    model.gamma12 = np.random.uniform(ci['gamma12'][1][0],ci['gamma12'][1][1],1)[0]
    model.gamma21 = np.random.uniform(ci['gamma21'][1][0],ci['gamma21'][1][1],1)[0]

    #print(model.E(d))
    bootstrap_samples[i,:] = model.E(df['drug1.conc'], df['drug2.conc']).to_numpy().flatten()

q_l = np.zeros(n_predict)
q_u = np.zeros(n_predict)

for j in range(n_predict):
    q_l[j] = np.nanquantile(bootstrap_samples[:,j], 0.025)
    q_u[j] = np.nanquantile(bootstrap_samples[:,j], 0.975)


df_ci = pd.DataFrame(np.concatenate([ci['beta'][1].reshape(1,2), ci['alpha12'][1].reshape(1,2), ci['alpha21'][1].reshape(1,2),
ci['gamma12'][1].reshape(1,2), ci['gamma21'][1].reshape(1,2)], axis=0)).round(2)
df_estimate = pd.DataFrame(np.concatenate([ci['beta'][0].reshape(1,1), ci['alpha12'][0].reshape(1,1), ci['alpha21'][0].reshape(1,1),
ci['gamma12'][0].reshape(1,1), ci['gamma21'][0].reshape(1,1)], axis=0)).round(2)

# Create data frame of the estimates of the parameters and confidence intervals to save in .csv file
df_result = pd.concat([df_estimate, df_ci], axis=1)
df_result.columns=['estimate', 'ci_l', 'ci_u']
df_result.index=['beta', 'alpha12', 'alpha21', 'gamma12', 'gamma21']
df_result.to_csv('results/LA_synergy/df_result_LA_synergy.csv')

#d11 = np.linspace(0.0, 14.0, num=100)
#d22 = np.linspace(0.0, 11.0, num=100)

[d11, d22] = np.meshgrid(np.linspace(0.1, A_max, num=100),  np.linspace(0.1, B_max, num=100))

Effect_musyc = model.E(d11, d22)
Effect_musyc = Effect_musyc.reshape((len(d11), len(d22)))


mean_full = model.E(df['drug1.conc'], df['drug2.conc']).to_numpy()

xyz_full = np.concatenate((df['drug1.conc'].to_numpy().reshape(-1,1), df['drug2.conc'].to_numpy().reshape(-1,1),mean_full.reshape(-1,1)),axis=1)

print(df)
#print(df['drug1.conc'].to_numpy().reshape(-1,1))
#print(df['drug2.conc'].to_numpy().reshape(-1,1))

xyz_full_lower = np.concatenate((df['drug1.conc'].to_numpy().reshape(-1,1), df['drug2.conc'].to_numpy().reshape(-1,1), q_l.reshape(-1,1)),axis=1)

xyz_full_upper = np.concatenate((df['drug1.conc'].to_numpy().reshape(-1,1), df['drug2.conc'].to_numpy().reshape(-1,1), q_u.reshape(-1,1)),axis=1)

Volume_full = trapezoidal_area(xyz_full)

Volume_full_lower = trapezoidal_area(xyz_full_lower)

Volume_full_upper = trapezoidal_area(xyz_full_upper)

print(Volume_full)
print(Volume_full_lower)
print(Volume_full_upper)

model.plot_surface_plotly(d11.flatten(), d22.flatten(), xlabel="Drug1", ylabel="Drug2", 				\
                          zlabel="Loewe Synergy", fname="plotly.html")

# plot
plt.figure(figsize=(12, 6))
plt.plot(dose_a.flatten(), effect_a.flatten(), "kx", mew=2)
plt.plot(np.linspace(0.1, A_max, num=100), Effect_musyc[0,:], "r", lw=2)
#plt.ylim((0.0, 1.0))
plt.title('Loewe synergy monotherapeutic slice of the musyc surface')
plt.savefig('figures/LA_synergy/Loewesynergy_musyc'+'drugA'+'.png')

# plot
plt.figure(figsize=(12, 6))
plt.plot(dose_b.flatten(), effect_b.flatten(), "kx", mew=2)
plt.plot(np.linspace(0.1, B_max, num=100), Effect_musyc[:,0], "r", lw=2)
#plt.ylim((0.0, 1.0))
plt.title('Loewe synergy monotherapeutic slice of the musyc surface')
plt.savefig('figures/LA_synergy/Loewesynergy_musyc'+'drugB'+'.png')

fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
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
plt.savefig('figures/LA_synergy/Difference_musyc_Loewesynergy.png')

fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(Dose_AA.flatten(), Dose_BB.flatten(), model.E(Dose_A, Dose_B).flatten().reshape(11,11))
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
#plt.title("MuSyC estimated surface", fontsize=20)
plt.savefig('figures/LA_synergy/Surface_musyc_Loewesynergy.png', bbox_inches = 'tight',
    pad_inches = 0)


# Null model MuSyC
model = MuSyC(variant='no_gamma',alpha12_bounds=(1,1),alpha21_bounds=(1,1))
model.fit(df['drug1.conc'], df['drug2.conc'], df['effect'],bootstrap_iterations=100)
ci_no_gamma = model.get_parameters()


print(ci_no_gamma)

mean_null = model.E(df['drug1.conc'], df['drug2.conc']).to_numpy()

xyz_null = np.concatenate((df['drug1.conc'].to_numpy().reshape(-1,1), df['drug2.conc'].to_numpy().reshape(-1,1),mean_null.reshape(-1,1)),axis=1)

print(df)
#print(df['drug1.conc'].to_numpy().reshape(-1,1))
#print(df['drug2.conc'].to_numpy().reshape(-1,1))
Volume_null = trapezoidal_area(xyz_null)

print('Volume difference', Volume_full-Volume_null)

exit()

fig, ax = plt.subplots(figsize=(6,6))
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
plt.savefig('figures/LA_synergy/Surface_null_musyc_LA_synergy.png', bbox_inches = 'tight',
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
plt.savefig('figures/LA_synergy/Difference_musyc_LA_synergy_2.png', bbox_inches = 'tight',
    pad_inches = 0)


print('Finished')
