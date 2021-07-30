from synergy.combination import MuSyC # or BRAID, Zimmer
from synergy.combination import Loewe, Bliss # or Bliss, ZIP, HSA, Schindler, CombinationIndex
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from synergy.utils import plots
import scipy

from synergy.combination.jacobians.musyc_jacobian import jacobian

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

'''
We fit MuSyC model on Greco data
'''

df = pd.read_csv('../data/main_greco_3times6.csv', sep=';')
df = df.astype(np.float64)
df_original = df

df = df.sort_values(by=['Dose1'])
df_original = df_original.sort_values(by=['Dose1'])

print(df)

Effect_AB = df['Response'].values.astype(np.float64).reshape(-1,1)
Effect = df['Response'].values.astype(np.float64).reshape(-1,1).copy()
Dose_A = df['Dose1'].values.astype(np.float64).copy()
Dose_B = df['Dose2'].values.astype(np.float64).copy()
Dose_AB = np.concatenate((Dose_A.reshape(-1,1), Dose_B.reshape(-1,1)), axis=1)
A_max = np.max(df['Dose1'].to_numpy())
B_max = np.max(df['Dose2'].to_numpy())

Dose_AA = df[df['Dose2']==0]['Dose1'].to_numpy().reshape(-1,1).astype(np.float64)
Dose_BB = df[df['Dose1']==0]['Dose2'].to_numpy().reshape(-1,1).astype(np.float64)

# One time point
df = pd.concat([df['Dose1'], df['Dose2'], df['Response']], axis=1)
df.columns=['drug1.conc', 'drug2.conc', 'effect']

single_dose_a = df[df['drug2.conc']==0]
single_dose_b = df[df['drug1.conc']==0]

dose_a = single_dose_a['drug1.conc'].to_numpy()
effect_a = single_dose_a['effect'].to_numpy()

dose_b = single_dose_b['drug2.conc'].to_numpy()
effect_b = single_dose_b['effect'].to_numpy()

'''
Define the unconstrained model
'''

model = MuSyC()
model = MuSyC(E1_bounds=(0.0,100.0),E2_bounds=(0.0,100.0), E3_bounds=(0.0,100.0))
# Optimize
model.fit(df['drug1.conc'], df['drug2.conc'], df['effect'], bootstrap_iterations=100, use_jacobian = False)

ci = model.get_parameters(confidence_interval=95)

# Predict effect with the optimized model
[d11, d22] = np.meshgrid(np.linspace(0.0, A_max, num=100),  np.linspace(0.0, B_max, num=100))

Effect_musyc = model.E(d11, d22)
Effect_musyc = Effect_musyc.reshape((len(d11), len(d22)))

# Get confidence intervals of the parameters
ci = model.get_parameters(confidence_interval=95)

df_ci = pd.DataFrame(np.concatenate([ci['beta'][1].reshape(1,2), ci['alpha12'][1].reshape(1,2), ci['alpha21'][1].reshape(1,2),
ci['gamma12'][1].reshape(1,2), ci['gamma21'][1].reshape(1,2)], axis=0)).round(2)
df_estimate = pd.DataFrame(np.concatenate([ci['beta'][0].reshape(1,1), ci['alpha12'][0].reshape(1,1), ci['alpha21'][0].reshape(1,1),
ci['gamma12'][0].reshape(1,1), ci['gamma21'][0].reshape(1,1)], axis=0)).round(2)

# Create data frame of the estimates of the parameters and confidence intervals to save in .csv file
df_result = pd.concat([df_estimate, df_ci], axis=1)
df_result.columns=['estimate', 'ci_l', 'ci_u']
df_result.index=['beta', 'alpha12', 'alpha21', 'gamma12', 'gamma21']
df_result.to_csv('results/Greco/df_result_greco.csv')

'''
Plot fits of the unconstrained model
'''
# plot monotherapeutic curve
plt.figure(figsize=(12, 6))
plt.plot(dose_a.flatten(), effect_a.flatten(), "kx", mew=2)
plt.plot(np.linspace(0.1, A_max, num=100), Effect_musyc[0,:], "purple", lw=2)
plt.title('Greco monotherapeutic slice of the musyc surface')
plt.savefig('figures/Greco_4by6/Greco_musyc2'+'drugA'+'.png')

# plot monotherapeutic curve
plt.figure(figsize=(12, 6))
plt.plot(dose_b.flatten(), effect_b.flatten(), "kx", mew=2)
plt.plot(np.linspace(0.1, B_max, num=100), Effect_musyc[:,0], "purple", lw=2)
plt.title('Greco monotherapeutic slice of the musyc surface')
plt.savefig('figures/Greco_4by6/Greco_musyc2'+'drugB'+'.png')

# Defining unique doses for plotting
Dose_A = np.unique(Dose_A)
Dose_B = np.unique(Dose_B)

df = df.iloc[2:]               # initial point at 0 has three observations
df['effect'].iloc[0] = 106.7   # average of these three points

mean_full = model.E(d11, d22)

n_boot = 500
n_predict = len(model.E(d11, d22).flatten())

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
    bootstrap_samples[i,:] = model.E(d11, d22).flatten()

q_l = np.zeros(n_predict)
q_u = np.zeros(n_predict)

for j in range(n_predict):
    q_l[j] = np.nanquantile(bootstrap_samples[:,j], 0.025)
    q_u[j] = np.nanquantile(bootstrap_samples[:,j], 0.975)

xyz_full = np.concatenate((d11.reshape(-1,1), d22.reshape(-1,1),mean_full.reshape(-1,1)),axis=1)

xyz_full_lower = np.concatenate((d11.reshape(-1,1), d22.reshape(-1,1), q_l.reshape(-1,1)),axis=1)

xyz_full_upper = np.concatenate((d11.reshape(-1,1), d22.reshape(-1,1), q_u.reshape(-1,1)),axis=1)

Volume_full = trapezoidal_area(xyz_full)

Volume_full_lower = trapezoidal_area(xyz_full_lower)

Volume_full_upper = trapezoidal_area(xyz_full_upper)

print('Volume_lower', Volume_full)
print('Volume_lower', Volume_full_lower)
print('Volume_upper', Volume_full_upper)

exit()

df_mean = pd.concat([pd.DataFrame(d11.reshape(-1,1)), pd.DataFrame(d22.reshape(-1,1)), pd.DataFrame(mean_full.reshape(-1,1)),
pd.DataFrame(q_l.reshape(-1,1)), pd.DataFrame(q_u.reshape(-1,1))], axis=1)
df_mean.columns=['drug1.conc', 'drug2.conc', 'predicted_effect', 'lower', 'upper']
print(df_mean)

single_dose_a_mean = df_mean[df_mean['drug2.conc']==0]
single_dose_b_mean = df_mean[df_mean['drug1.conc']==0]

dose_a_mean = single_dose_a_mean['drug1.conc'].to_numpy()
effect_a_mean = single_dose_a_mean['predicted_effect'].to_numpy()
q_l_a_mean = single_dose_a_mean['lower'].to_numpy()
q_u_a_mean = single_dose_a_mean['upper'].to_numpy()

dose_b_mean = single_dose_b_mean['drug2.conc'].to_numpy()
effect_b_mean = single_dose_b_mean['predicted_effect'].to_numpy()
q_l_b_mean = single_dose_b_mean['lower'].to_numpy()
q_u_b_mean = single_dose_b_mean['upper'].to_numpy()

df_A = pd.concat([pd.DataFrame(dose_a_mean.reshape(-1,1)), pd.DataFrame(effect_a_mean.reshape(-1,1)), pd.DataFrame(q_l_a_mean),
pd.DataFrame(q_u_a_mean)], axis=1)
df_A.columns=['dose', 'predicted_effect', 'lower', 'upper']

df_A = df_A.sort_values(by=['dose'])

df_B = pd.concat([pd.DataFrame(dose_b_mean.reshape(-1,1)), pd.DataFrame(effect_b_mean.reshape(-1,1)), pd.DataFrame(q_l_b_mean),
pd.DataFrame(q_u_b_mean)], axis=1)
df_B.columns=['dose', 'predicted_effect', 'lower', 'upper']

df_B = df_B.sort_values(by=['dose'])

# plot monotherapeutic curve
plt.figure(figsize=(12, 10))
plt.plot(dose_a[2:6].flatten(), effect_a[2:6].flatten(), "kx", mew=2)
plt.plot(df_A['dose'], df_A['predicted_effect'],"C0", lw=2, color='purple')
plt.fill_between(
    df_A['dose'],
    df_A['lower'],
    df_A['upper'],
    color="purple",
    alpha=0.2
)
plt.xlabel('$x_1$', fontsize=40)
plt.ylabel('Response', fontsize=40)
#plt.title('Monotherapeutic slice of the MuSyC surface', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=40)
#plt.title('Greco monotherapeutic slice of the musyc surface')
plt.savefig('figures/Greco_4by6/Greco_musyc2'+'drugA'+'.png', bbox_inches="tight")

# plot monotherapeutic curve
plt.figure(figsize=(12, 10))
plt.plot(dose_b[2:8].flatten(), effect_b[2:8].flatten(), "kx", mew=2)
plt.plot(df_B['dose'], df_B['predicted_effect'],"C0", lw=2, color='purple')
plt.fill_between(
    df_B['dose'],
    df_B['lower'],
    df_B['upper'],
    color="purple",
    alpha=0.2
)
plt.xlabel('$x_2$', fontsize=40)
plt.ylabel('Response', fontsize=40)
#plt.title('Monotherapeutic slice of the MuSyC surface', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=40)
#plt.title('Greco monotherapeutic slice of the musyc surface')
plt.savefig('figures/Greco_4by6/Greco_musyc2'+'drugB'+'.png', bbox_inches="tight")

'''
Plot residuals
'''
fig, ax = plt.subplots(figsize=(6,6))
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
#v = np.linspace(-8, 15, 10, endpoint=True)
cf = ax.contourf(Dose_A, Dose_B, (df['effect'].to_numpy()-model.E(df['drug1.conc'], df['drug2.conc']).to_numpy() ).flatten().reshape(6,4), cmap='YlGnBu')
cbar = fig.colorbar(cf, ax=ax)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)
#plt.title("MuSyC residuals", fontsize=20)
plt.savefig('figures/Greco_4by6/Difference_musyc_Greco.png', bbox_inches = 'tight',
    pad_inches = 0)

'''
Plot MuSyC surface
'''
fig, ax = plt.subplots(figsize=(6,6))
#ax.set_aspect('equal')
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
v = np.linspace(0.0, 120.0, 10, endpoint=True)
cf = ax.contourf(Dose_A, Dose_B, model.E(df['drug1.conc'], df['drug2.conc']).to_numpy().reshape(6,4),v)
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
    t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)
#plt.title("MuSyC surface", fontsize=20)
plt.savefig('figures/Greco_4by6/Surface_musyc_Greco.png', bbox_inches = 'tight',
    pad_inches = 0)

result = np.concatenate((np.asarray(df['drug1.conc']).reshape(-1,1), np.asarray(df['drug2.conc']).reshape(-1,1), np.asarray(model.E(df['drug1.conc'], df['drug2.conc']) - df['effect']).reshape(-1,1)), axis = 1)

'''
Define and optimize constrained (null) MuSyC model
'''

Effect_AB = df['effect'].values.reshape(-1,1)
model = MuSyC(variant='no_gamma',E1_bounds=(0.0000001,100.0),E2_bounds=(0.0000001,100.0), E3_bounds=(0.0000001,100.0),C1_bounds=(0.00001,100.0),alpha12_bounds=(1,1),alpha21_bounds=(1,1))

model.fit(df['drug1.conc'], df['drug2.conc'], df['effect'], bootstrap_iterations=100, use_jacobian = False)
ci_no_gamma = model.get_parameters(confidence_interval=95)
mean_null = model.E(d11, d22)

'''
Plot null surface
'''
fig, ax = plt.subplots(figsize=(6,6))
#v = np.linspace(0.0, 120.0, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(Dose_A, Dose_B, model.E(df['drug1.conc'], df['drug2.conc']).to_numpy().reshape(6,4))
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.savefig('figures/Greco_4by6/Surface_null_musyc_Greco.png', bbox_inches = 'tight',
    pad_inches = 0)

'''
Plot difference between null surface and the effect
'''
fig, ax = plt.subplots(figsize=(6,6))
v = np.linspace(-60, 30, 11, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(Dose_A, Dose_B, (model.E(df['drug1.conc'], df['drug2.conc']).to_numpy()-Effect_AB.flatten()).reshape(6,4), v, cmap='RdYlGn')
cbar = fig.colorbar(cf, ax=ax)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
#plt.title("MuSyC residuals", fontsize=20)
plt.savefig('figures/Greco_4by6/Difference_musyc_Greco_2.png', bbox_inches = 'tight',
    pad_inches = 0)

#print('MSE total',np.sum((df['effect'].to_numpy()-model.E(df['drug1.conc'], df['drug2.conc']).to_numpy() ).flatten()**2)/(len(df['effect'].to_numpy())))

'''
Plot monotherapeutic results
'''
single_dose_a = df[df['drug2.conc']==0]
single_dose_b = df[df['drug1.conc']==0]

dose_a = single_dose_a['drug1.conc'].to_numpy()
effect_a = single_dose_a['effect'].to_numpy()

dose_b = single_dose_b['drug2.conc'].to_numpy()
effect_b = single_dose_b['effect'].to_numpy()


Effect = df_original['Response'].values.reshape(-1,1).copy()
Dose_A = df_original['Dose1'].values.astype(np.float64).copy()
Dose_B = df_original['Dose2'].values.astype(np.float64).copy()
Dose_AB = np.concatenate((Dose_A.reshape(-1,1), Dose_B.reshape(-1,1)), axis=1)

Effect_B = df_original[df_original['Dose1'] == 0]['Response'].to_numpy().reshape(-1,1).astype(np.float64)
Effect_A = df_original[df_original['Dose2'] == 0]['Response'].to_numpy().reshape(-1,1).astype(np.float64)
Dose_A = df_original[df_original['Dose2']==0]['Dose1'].to_numpy().reshape(-1,1).astype(np.float64)
Dose_B = df_original[df_original['Dose1']==0]['Dose2'].to_numpy().reshape(-1,1).astype(np.float64)

# plot
plt.figure(figsize=(12, 10))
plt.plot(Dose_A.flatten(), Effect_A.flatten(), "kx", mew=2)
plt.plot(np.linspace(0.0, A_max, num=100), Effect_musyc[0,:], "purple", lw=2)
#plt.ylim((0.0, 1.0))
plt.xlabel('$x_1$', fontsize=40)
plt.ylabel('Response', fontsize=40)
#plt.title('Monotherapeutic slice of the MuSyC surface', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=40)
plt.savefig('figures/Greco_4by6/Greco'+'drugA'+'.png', bbox_inches="tight")

# plot
plt.figure(figsize=(12, 10))
plt.plot(Dose_B.flatten(), Effect_B.flatten(), "kx", mew=2)
plt.plot(np.linspace(0.0, B_max, num=100), Effect_musyc[:,0], "purple", lw=2)
#plt.ylim((0.0, 1.0))
plt.xlabel('$x_2$', fontsize=40)
plt.ylabel('Response', fontsize=40)
#plt.title('Monotherapeutic slice of the MuSyC surface', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=40)
plt.savefig('figures/Greco_4by6/Greco'+'drugB'+'.png', bbox_inches="tight")

# plot
plt.figure(figsize=(12, 10))
plt.plot(np.log(np.delete(Dose_A.flatten(),[0,1,2])), np.delete(Effect_A.flatten(),[0,1,2]), "kx", mew=2)
plt.plot(np.log(np.linspace(0.1, A_max, num=100)), Effect_musyc[0,:], "purple", lw=2)
#plt.ylim((0.0, 1.0))
plt.xlabel('$log(x_1)$', fontsize=40)
plt.ylabel('Response', fontsize=40)
#plt.title('Monotherapeutic slice of the MuSyC surface', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=40)
plt.savefig('figures/Greco_4by6/Greco'+'drugA_in_log'+'.png', bbox_inches="tight")

# plot
plt.figure(figsize=(12, 10))
plt.plot(np.log(np.delete(Dose_B.flatten(),[0,1,2])), np.delete(Effect_B.flatten(),[0,1,2]), "kx", mew=2)
plt.plot(np.log(np.linspace(0.1, B_max, num=100)), Effect_musyc[:,0], "purple", lw=2)
#plt.ylim((0.0, 1.0))
plt.xlabel('$log(x_2)$', fontsize=40)
plt.ylabel('Response', fontsize=40)
#plt.title('Monotherapeutic slice of the MuSyC surface', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=40)
plt.savefig('Greco'+'drugB_in_log'+'.png', bbox_inches="tight")

print('Finished')
