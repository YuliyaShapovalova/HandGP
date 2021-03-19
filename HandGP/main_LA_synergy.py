import numpy as np
from numpy.linalg import inv
from bisect import bisect_left
from scipy import special
from scipy.special import erf
import scipy.linalg
from scipy.stats.distributions import chi2
from scipy.interpolate import interp1d
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from gpflow.ci_utils import ci_niter
from gpflow import set_trainable
from gpflow.utilities import print_summary, positive
from gpflow import set_trainable
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pymc3

np.set_printoptions(suppress=True)

from utilities import (compute_prior_hyperparameters, trapezoidal_area, predict_in_observations, fit_Hand, y_exp_Hand, K_multiplicative)

f64 = gpflow.utilities.to_default_float

drug_pair = 'LA_synergy'
drug_name = 'LA_synergy'

df = pd.read_csv('../data/'+str(drug_pair)+'.csv')

Effect = df['E'].values.reshape(-1,1).copy()
Dose_A = np.asarray(df['DRUG_A'].values.astype(float).copy()).reshape(-1,1)
Dose_B = np.asarray(df['DRUG_B'].values.astype(float).copy()).reshape(-1,1)
Dose_AB = np.concatenate((Dose_A.reshape(-1,1), Dose_B.reshape(-1,1)), axis=1)

Effect_B = df[df['DRUG_A'] == 0]['E'].to_numpy().reshape(-1,1).astype(float)
Effect_A = df[df['DRUG_B'] == 0]['E'].to_numpy().reshape(-1,1).astype(float)
Dose_A = df[df['DRUG_B']==0]['DRUG_A'].to_numpy().reshape(-1,1).astype(float)
Dose_B = df[df['DRUG_A']==0]['DRUG_B'].to_numpy().reshape(-1,1).astype(float)

# set hyperparameters
A_max  = np.max(Dose_A)
B_max = np.max(Dose_B)


alphaA, betaA = compute_prior_hyperparameters(A_max/5, 0.01*A_max)
alphaB, betaB = compute_prior_hyperparameters(B_max/5, 0.01*B_max)


eff_max_a = np.max(Effect_A)
eff_max_b = np.max(Effect_B)
eff_max = np.max([eff_max_a, eff_max_b])

alpha_var, beta_var = compute_prior_hyperparameters(eff_max/2, 0.01*eff_max)

# Add extra monotherapeutic data
zeros_A = np.zeros((Dose_A.shape))
zeros_B = np.zeros((Dose_B.shape))

Dose_A_mono = np.concatenate((Dose_A.reshape(-1,1), zeros_A.reshape(-1,1)),  axis=0)
Dose_B_mono = np.concatenate((zeros_B.reshape(-1,1), Dose_B.reshape(-1,1)),  axis=0)

Dose_AB_mono = np.concatenate((Dose_A_mono.reshape(-1,1), Dose_B_mono.reshape(-1,1)),  axis=1)
Effect_mono = np.concatenate((Effect_A.reshape(-1,1), Effect_B.reshape(-1,1)),  axis=0)

Dose_AB = np.concatenate((Dose_AB, Dose_AB_mono, Dose_AB_mono, Dose_AB_mono, Dose_AB_mono),  axis=0)
Effect = np.concatenate((Effect.reshape(-1,1), Effect_mono.reshape(-1,1), Effect_mono.reshape(-1,1), Effect_mono.reshape(-1,1), Effect_mono.reshape(-1,1)),  axis=0)

'''
Now we set-up the model in gpflow and optimize it;
we start with different initial values for the hyperparameters of the kernels.
'''

[l1_init, l2_init] = np.meshgrid(np.linspace(.1, 105.0, 10), np.linspace(.1, 105.0, 10))
l1_init = l1_init.reshape(-1,1)
l2_init = l2_init.reshape(-1,1)
Lik_null = np.zeros((100,1))
Lik_full = np.zeros((100,1))
var_init =  np.zeros((100,1))

for i in range(0,100):
    try:
        init_lengthscale_da = l1_init[i,0]
        init_lengthscale_db = l2_init[i,0]
        init_variance = eff_max
        init_likelihood_variance = 0.01

        k = K_multiplicative()

        m = gpflow.models.GPR(data=(Dose_AB, Effect), kernel=k, mean_function=None)

        m.likelihood.variance.assign(0.01)
        m.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(0.14), np.float64(1.14))

        m.kernel.lengthscale_da.assign(init_lengthscale_da)
        m.kernel.lengthscale_db.assign(init_lengthscale_db)
        m.kernel.variance_da.assign(alpha_var)

        m.kernel.variance_da.prior = tfp.distributions.Gamma(np.float64(alpha_var), np.float64(beta_var))
        m.kernel.lengthscale_da.prior = tfp.distributions.Gamma(np.float64(alphaA), np.float64(betaA))
        m.kernel.lengthscale_db.prior = tfp.distributions.Gamma(np.float64(alphaB), np.float64(betaB))

        opt = gpflow.optimizers.Scipy()

        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
        print_summary(m)

        Lik_full[i,0] = np.asarray(m.training_loss())
        var_init[i,0] = np.asarray(m.kernel.variance_da.value())

    except:
        Lik_full[i,0] = 'NaN'
        print('Cholesky was not successful')

index = np.where(Lik_full == np.nanmin(Lik_full))[0][0]

init_lengthscale_da = l1_init[index,0]
init_lengthscale_db = l2_init[index,0]
init_var = var_init[index,0]

m = gpflow.models.GPR(data=(Dose_AB, Effect), kernel=k, mean_function=None)

m.likelihood.variance.assign(0.01)
m.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(0.14), np.float64(1.14))

m.kernel.lengthscale_da.assign(init_lengthscale_da)
m.kernel.lengthscale_db.assign(init_lengthscale_db)
m.kernel.variance_da.assign(init_var)

# priors
m.kernel.variance_da.prior = tfp.distributions.Gamma(np.float64(alpha_var), np.float64(beta_var))
m.kernel.lengthscale_da.prior = tfp.distributions.Gamma(np.float64(alphaA), np.float64(betaA))
m.kernel.lengthscale_db.prior = tfp.distributions.Gamma(np.float64(alphaB), np.float64(betaB))

opt = gpflow.optimizers.Scipy()

opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

print_summary(m)

# Predict and plot monotherameutic curves
[Xi, Xj] = np.meshgrid(np.linspace(0.0, np.max(Dose_A), 50), np.linspace(0.0, np.max(Dose_B), 50))

# We need to augument our test space to be a list of coordinates for input to the GP
Xnew2 = np.vstack((Xi.ravel(), Xj.ravel())).T # Change our input grid to list of coordinates

# Predict the mean and covariance of the GP fit at the test locations
mean2, Cov2 = m.predict_f(Xnew2)

mean2 = np.asarray(mean2)
Cov2 = np.asarray(Cov2)

num_predict = 50

xx_A = np.linspace(0.0, np.max(Dose_A),  num_predict).reshape(num_predict, 1)
xx_B = np.linspace(0.0, np.max(Dose_B),  num_predict).reshape( num_predict, 1)

mean2 = pd.DataFrame(np.asarray(mean2).reshape(Xi.shape))
Cov2 = pd.DataFrame(np.asarray(Cov2).reshape(Xi.shape))

## plot
plt.figure(figsize=(12, 6))
plt.plot(Dose_A, Effect_A, "kx", mew=2)
plt.plot(xx_A, mean2.loc[0],"C0", lw=2, color='red')
plt.fill_between(
    xx_A[:, 0],
    mean2.loc[0] - 1.96 * np.sqrt(Cov2.loc[0]),
    mean2.loc[0] + 1.96 * np.sqrt( Cov2.loc[0]),
    color="red",
    alpha=0.2
)
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('Response', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('figures/LA_synergy/'+drug_name+'_DrugA_2drugs'+'.png')

## plot
plt.figure(figsize=(12, 6))
plt.plot(Dose_B, Effect_B, "kx", mew=2)
plt.plot(xx_B, mean2.iloc[:,0],"C0", lw=2, color='red')
plt.fill_between(
    xx_B[:, 0],
    mean2.iloc[:,0] - 1.96 * np.sqrt(Cov2.iloc[:,0]),
    mean2.iloc[:,0] + 1.96 * np.sqrt( Cov2.iloc[:,0]),
    color="red",
    alpha=0.2
)
plt.xlabel('$x_2$', fontsize=20)
plt.ylabel('Response', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('figures/LA_synergy/'+drug_name+'_DrugB_2drugs'+'.png')


dim2_A = mean2.iloc[0].to_numpy()
dim2_B = mean2.loc[:][0].to_numpy()

N = 5000

Effect = df['E'].values.reshape(-1,1).copy()
Dose_A = df['DRUG_A'].values.astype(float).copy()
Dose_B = df['DRUG_B'].values.astype(float).copy()
Dose_AB = np.concatenate((Dose_A.reshape(-1,1), Dose_B.reshape(-1,1)),  axis=1)

Effect_B = df[df['DRUG_A'] == 0]['E'].to_numpy().reshape(-1,1).astype(float)
Effect_A = df[df['DRUG_B'] == 0]['E'].to_numpy().reshape(-1,1).astype(float)
Dose_A = df[df['DRUG_B']==0]['DRUG_A'].to_numpy().reshape(-1,1).astype(float)
Dose_B = df[df['DRUG_A']==0]['DRUG_B'].to_numpy().reshape(-1,1).astype(float)

data = pd.concat([pd.DataFrame(Dose_AB), pd.DataFrame(Effect)], axis=1)
data.columns = ['Dose_A', 'Dose_B', 'Effect']

X1 = data[data['Dose_B']==0]['Dose_A'].to_numpy().reshape(-1,1).astype(float)
Y1 = data[data['Dose_B']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

X2 = data[data['Dose_A']==0]['Dose_B'].to_numpy().reshape(-1,1).astype(float)
Y2 = data[data['Dose_A']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

Y_expected_Hand = fit_Hand(X1, X2, dim2_A, dim2_B, Dose_A, Dose_B)
Y_expected_Hand_check = fit_Hand(X2, X1, dim2_B, dim2_A, Dose_B, Dose_A)

mean_full, Cov_full = predict_in_observations(X1, X2, m)

xv, yv = np.meshgrid(X1, X2)

[Xi, Xj] = np.meshgrid(X1,X2)

xyz_full = np.concatenate((Xi.reshape(-1,1), Xj.reshape(-1,1), mean_full.reshape(-1,1)),axis=1)
Volume_full = trapezoidal_area(xyz_full)


xyz_null = np.concatenate((Xi.reshape(-1,1), Xj.reshape(-1,1), Y_expected_Hand.reshape(-1,1)),axis=1)
Volume_null = trapezoidal_area(xyz_null)

print('Volume difference', Volume_full-Volume_null)

xx_a = np.linspace(np.min(Dose_A), np.max(Dose_A), dim2_A.shape[0]).reshape(-1,1)
xx_b = np.linspace(np.min(Dose_B), np.max(Dose_B), dim2_B.shape[0]).reshape(-1,1)

fig, ax = plt.subplots(figsize=(6,6))
v = np.linspace(-0.15, 1.05, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(np.log(Dose_A.flatten()/np.array(m.kernel.lengthscale_da.value())+1), np.log(Dose_B.flatten()/np.array(m.kernel.lengthscale_db.value())+1),Y_expected_Hand,v)
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)

plt.xlabel('log($x_1$/$l_1$+1)', fontsize=15)
plt.ylabel('log($x_2$/$l_2$+1)', fontsize=15)
plt.savefig('figures/LA_synergy/'+str(drug_name)+'_Hand_contour_result'+'.png', bbox_inches = 'tight',
    pad_inches = 0)

fig, ax = plt.subplots(figsize=(6,6))
v = np.linspace(0.0, 1.05, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(np.log(Dose_A.flatten()/np.array(m.kernel.lengthscale_da.value())+1), np.log(Dose_B.flatten()/np.array(m.kernel.lengthscale_db.value())+1),mean_full)
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)

plt.xlabel('log($x_1$/$l_1$+1)', fontsize=15)
plt.ylabel('log($x_2$/$l_2$+1)', fontsize=15)
plt.savefig('figures/LA_synergy/'+str(drug_name)+'_GP_mean'+'.png', bbox_inches = 'tight',
    pad_inches = 0)

fig, ax = plt.subplots(figsize=(6,6))
v = np.linspace(-1.0, 1.0, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(np.log(Dose_A.flatten()/np.array(m.kernel.lengthscale_da.value())+1), np.log(Dose_B.flatten()/np.array(m.kernel.lengthscale_db.value())+1),mean_full - Y_expected_Hand,v,cmap='RdYlGn')
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)

plt.xlabel('log($x_1$/$l_1$+1)', fontsize=15)
plt.ylabel('log($x_2$/$l_2$+1)', fontsize=15)
plt.savefig('figures/LA_synergy/'+str(drug_name)+'contour_result'+'.png', bbox_inches = 'tight',
    pad_inches = 0)

fig, ax = plt.subplots(figsize=(6,6))
v = np.linspace(-0.1,0.1, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(np.log(Dose_A.flatten()/np.array(m.kernel.lengthscale_da.value())+1), np.log(Dose_B.flatten()/np.array(m.kernel.lengthscale_db.value())+1),mean_full - Effect.reshape(11,11), v, cmap='YlGnBu')

cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)

plt.xlabel('log($x_1$/$l_1$+1)', fontsize=15)
plt.ylabel('log($x_2$/$l_2$+1)', fontsize=15)
#plt.title("GP residuals", fontsize=20)
plt.savefig('figures/LA_synergy/'+str(drug_name)+'_GP_residuals'+'.png', bbox_inches = 'tight',
    pad_inches = 0)

dim2_A = mean2.iloc[0].to_numpy()
dim2_B = mean2.loc[:][0].to_numpy()

data = pd.concat([pd.DataFrame(Dose_AB), pd.DataFrame(Effect)], axis=1)
data.columns = ['Dose_A', 'Dose_B', 'Effect']

X1 = data[data['Dose_B']==0]['Dose_A'].to_numpy().reshape(-1,1).astype(float)
Y1 = data[data['Dose_B']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

X2 = data[data['Dose_A']==0]['Dose_B'].to_numpy().reshape(-1,1).astype(float)
Y2 = data[data['Dose_A']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)
############################################################################
mean1 =  dim2_A.copy().reshape(-1,1)
mean2 =  dim2_B.copy().reshape(-1,1)
xx1 = np.linspace(np.min(Dose_A), np.max(Dose_A), mean1.shape[0]).reshape(-1,1)
xx2 = np.linspace(np.min(Dose_B), np.max(Dose_B), mean2.shape[0]).reshape(-1,1)

#d1 = {'fa': mean1, 'fb': mean2, 'xx_a': xx1, 'xx_b': xx2}
d1 = np.concatenate((mean1, mean2, xx1, xx2), axis=1)
df1 = pd.DataFrame(data=d1)
df1.columns = ['fa', 'fb', 'xx_a', 'xx_b']
df1.to_csv('../data/TestData'+str(drug_name)+'_new'+'.csv')

d2 = np.concatenate((X1, X2, Dose_A, Dose_B), axis=1)
#d2 = {'X1': [X1], 'X2': [X2], 'Dose_A': [Dose_A], 'Dose_B': [Dose_B]}
df2 = pd.DataFrame(data=d2)
df2.columns = ['X1', 'X2', 'Dose_A', 'Dose_B']
df2.to_csv('../data/TestData2'+str(drug_name)+'_new'+'.csv')
############################################################################

df1 = pd.read_csv('../data/TestData2LA_synergy.csv', sep= ",")
df2 = pd.read_csv('../data/TestDataLA_synergy.csv', sep= ",")

mean1 = np.asarray(df2['fa']).reshape(-1,1)
mean2 = np.asarray(df2['fb']).reshape(-1,1)

xx1 = np.asarray(df2['xx_a']).reshape(-1,1)
xx2 = np.asarray(df2['xx_b']).reshape(-1,1)

X1 = np.asarray(df1['X1']).reshape(-1,1)
X2 = np.asarray(df1['X2']).reshape(-1,1)

Dose_A = np.asarray(df1['Dose_A']).reshape(-1,1)
Dose_B = np.asarray(df1['Dose_B']).reshape(-1,1)

a = 25.0
b = 25.0

Y_expected_Hand = fit_Hand(X1, X2, mean1, mean2, Dose_A, Dose_B)

fig, ax = plt.subplots(figsize=(6,6))
v = np.linspace(-1.0, 1.0, 10, endpoint=True)
#ax.set_aspect('equal')
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
#cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(), mean_full - Y_expected_Hand, v, cmap='RdYlGn')
cf = ax.contourf(np.log(Dose_A.flatten()/np.array(m.kernel.lengthscale_da.value())+1), np.log(Dose_B.flatten()/np.array(m.kernel.lengthscale_db.value())+1),mean_full - Y_expected_Hand,v, cmap='RdYlGn')
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.xlabel('log($x_1$/$l_1$+1)', fontsize=15)
plt.ylabel('log($x_2$/$l_2$+1)', fontsize=15)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.savefig('figures/LA_synergy/'+drug_name+'contour_result'+'.png', bbox_inches = 'tight',
    pad_inches = 0)

fig, ax = plt.subplots(figsize=(6,6))
v = np.linspace(-0.15, 1.05, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(np.log(Dose_A.flatten()/np.array(m.kernel.lengthscale_da.value())+1), np.log(Dose_B.flatten()/np.array(m.kernel.lengthscale_db.value())+1),Y_expected_Hand,v)
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('log($x_1$/$l_1$+1)', fontsize=15)
plt.ylabel('log($x_2$/$l_2$+1)', fontsize=15)
plt.savefig('figures/LA_synergy/'+drug_name+'_Hand_contour_result'+'.png', bbox_inches = 'tight',
    pad_inches = 0)

fig, ax = plt.subplots(figsize=(6,6))
v = np.linspace(-0.1, 0.1, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(np.log(Dose_A.flatten()/np.array(m.kernel.lengthscale_da.value())+1), np.log(Dose_B.flatten()/np.array(m.kernel.lengthscale_db.value())+1),mean_full - Effect.reshape(11,11), v, cmap='YlGnBu')
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)

plt.xlabel('log($x_1$/$l_1$+1)', fontsize=15)
plt.ylabel('log($x_2$/$l_2$+1)', fontsize=15)

plt.savefig('figures/LA_synergy/'+drug_name+'_GP_residuals'+'.png', bbox_inches = 'tight',
    pad_inches = 0)

def plot_samples(step_size, num_leapfrog, samples, parameters, y_axis_label):
    fig, axes = plt.subplots(1, len(param_to_name), figsize=(15, 3), constrained_layout=True)
    for ax, val, param in zip(axes, samples, parameters):
        ax.plot(np.stack(val).flatten())
        ax.set_title(param_to_name[param])
    fig.suptitle(y_axis_label)
    plt.savefig('figures/LA_synergy/LA_synergy_samples_'+str(step_size)+str(num_leapfrog)+'.png')


def marginal_samples(step_size, leap_frog, samples, parameters, y_axis_label):
    fig, axes = plt.subplots(1, len(param_to_name), figsize=(15, 3), constrained_layout=True)
    for ax, val, param in zip(axes, samples, parameters):
        ax.hist(np.stack(val).flatten(), bins=20)
        ax.set_title(param_to_name[param])
    fig.suptitle(y_axis_label)
    plt.savefig('figures/LA_synergy/LA_synergy_samples_'+str(step_size)+str(leap_frog)+'.png')

    res = np.zeros((4, 2))
    for i in range(3):
        hpd = pymc3.stats.hpd(np.asarray(samples[i]))
        res[i,0] = hpd[0]
        res[i,1] = hpd[1]
    print(res)

    res = np.concatenate((np.array(['l_a', 'l_b', 'var', 'var_lik']).reshape(-1,1), np.array([m.kernel.lengthscale_da.value(),
    m.kernel.lengthscale_db.value(), m.kernel.variance_da.value(), m.likelihood.variance.value()]).reshape(-1,1), res), axis=1)

    df = pd.DataFrame(res)
    print(df)

    df.columns = ['parameter','MAP','hpd_l', 'hpd_u']

    df['MAP'] = df['MAP'].astype(float).round(2)
    df['hpd_l'] = df['hpd_l'].astype(float).round(2)
    df['hpd_u'] = df['hpd_u'].astype(float).round(2)

    print(df)
    df = df.round(2)
    df.to_csv('results/LA_synergy/LA_synergy_hyperparameters'+str(step_size)+str(leap_frog)+'.csv')

'''
Run HMC.
General strategy is to run different combinations of the step size and leap frog
step and pick the largest step size with the highest acceptance probability.
'''

#leapfrog_num_and_step_size = np.stack(np.meshgrid([20.0, 15.0, 10.0, 5.0, 1.0], [0.1, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0])).T.reshape(-1, 2)
leapfrog_num_and_step_size = np.stack(np.meshgrid([ 5.0], [ 0.1])).T.reshape(-1, 2)
num_hmc_parameters = leapfrog_num_and_step_size.shape[0]

for j in range(num_hmc_parameters):
    #try:

    num_burnin_steps = ci_niter(1000)
    num_samples = ci_niter(100000)

    num_leapfrog, step_size = leapfrog_num_and_step_size[j]

    # Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
    hmc_helper = gpflow.optimizers.SamplingHelper(
        m.log_posterior_density, m.trainable_parameters
    )

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=num_leapfrog, step_size=step_size
    )
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
    )


    @tf.function
    def run_chain_fn():
        return tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            current_state=hmc_helper.current_state,
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        )


    samples, traces = run_chain_fn()
    parameter_samples = hmc_helper.convert_to_constrained_values(samples)

    param_to_name = {param: name for name, param in gpflow.utilities.parameter_dict(m).items()}

    #marginal_samples(step_size,samples, m_full.trainable_parameters, "unconstrained variable samples")
    marginal_samples(step_size,num_leapfrog,parameter_samples, m.trainable_parameters, "constrained parameter samples")

    #plot_samples(samples, m_full.trainable_parameters, "unconstrained values")
    plot_samples(step_size,num_leapfrog,parameter_samples, m.trainable_parameters, "constrained parameter samples")

    #except:
    #    print('HMC did not work')
