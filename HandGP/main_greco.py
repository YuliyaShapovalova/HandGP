import numpy as np
import tensorflow_probability as tfp
import pandas as pd
from bisect import bisect_left
from scipy import special
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.linalg import inv
import tensorflow as tf
import scipy.linalg
import gpflow
from gpflow.ci_utils import ci_niter
from gpflow import set_trainable
from scipy.stats.distributions import chi2
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from gpflow.utilities import print_summary, positive
np.set_printoptions(suppress=True)
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pymc3

from utilities import (predict_in_observations_lower, predict_in_observations_upper, trapezoidal_area, compute_prior_hyperparameters,  predict_in_observations, fit_Hand, y_exp_Hand, y_exp, find_nearest, K_log, K_multiplicative)

f64 = gpflow.utilities.to_default_float

df = pd.read_csv('../data/GrecoSimulatedData.csv', sep=';')
df = df.sort_values(by=['Dose1','Dose2'])
Effect = df['Response'].values.reshape(-1,1).copy()
Dose_A = df['Dose1'].values.astype(float).copy()
Dose_B = df['Dose2'].values.astype(float).copy()
Dose_AB = np.concatenate((Dose_A.reshape(-1,1), Dose_B.reshape(-1,1)), axis=1)

Effect_B = df[df['Dose1'] == 0]['Response'].to_numpy().reshape(-1,1).astype(float)
Effect_A = df[df['Dose2'] == 0]['Response'].to_numpy().reshape(-1,1).astype(float)
Dose_A = df[df['Dose2']==0]['Dose1'].to_numpy().reshape(-1,1).astype(float)
Dose_B = df[df['Dose1']==0]['Dose2'].to_numpy().reshape(-1,1).astype(float)

# hyperparameters of the priors
A_max  = np.max(Dose_A)
B_max = np.max(Dose_B)

A_min  = np.min(Dose_A)
B_min = np.min(Dose_B)

eff_max_a = np.max(Effect_A)
eff_max_b = np.max(Effect_B)

eff_min_a = np.min(Effect_A)
eff_min_b = np.min(Effect_B)

eff_max = np.max(Effect)

print(eff_max)

c_a = eff_max_a/eff_min_a
c_b = eff_max_b /eff_min_b


alphaA, betaA = compute_prior_hyperparameters(A_max/c_a, 0.1*A_max/c_a)
alphaB, betaB = compute_prior_hyperparameters(B_max/c_b, 0.1*B_max/c_b)

alpha_var, beta_var = compute_prior_hyperparameters(eff_max, 0.1*eff_max)

data = pd.concat([pd.DataFrame(Dose_AB), pd.DataFrame(Effect)], axis=1)
data.columns = ['Dose_A', 'Dose_B', 'Effect']

X1 = data[data['Dose_B']==0]['Dose_A'].to_numpy().reshape(-1,1).astype(float)
Y1 = data[data['Dose_B']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

X2 = data[data['Dose_A']==0]['Dose_B'].to_numpy().reshape(-1,1).astype(float)
Y2 = data[data['Dose_A']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

Dose_A = data[data['Dose_B']==0]['Dose_A'].to_numpy().reshape(-1,1).astype(float)
Effect_A = data[data['Dose_B']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

Dose_B = data[data['Dose_A']==0]['Dose_B'].to_numpy().reshape(-1,1).astype(float)
Effect_B = data[data['Dose_A']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)


zeros_A = np.zeros((Dose_A.shape))
zeros_B = np.zeros((Dose_B.shape))

Dose_A_mono = np.concatenate((Dose_A.reshape(-1,1), zeros_A.reshape(-1,1)),  axis=0)
Dose_B_mono = np.concatenate((zeros_B.reshape(-1,1), Dose_B.reshape(-1,1)),  axis=0)

Dose_AB_mono = np.concatenate((Dose_A_mono.reshape(-1,1), Dose_B_mono.reshape(-1,1)),  axis=1)
Effect_mono = np.concatenate((Effect_A.reshape(-1,1), Effect_B.reshape(-1,1)),  axis=0)

Dose_AB = np.concatenate((Dose_AB, Dose_AB_mono, Dose_AB_mono, Dose_AB_mono, Dose_AB_mono),  axis=0)
Effect = np.concatenate((Effect.reshape(-1,1), Effect_mono.reshape(-1,1), Effect_mono.reshape(-1,1), Effect_mono.reshape(-1,1), Effect_mono.reshape(-1,1)),  axis=0)

[l1_init, l2_init] = np.meshgrid(np.linspace(0.01, np.max(Dose_A), 10), np.linspace(0.01,  np.max(Dose_B), 10))
l1_init = l1_init.reshape(-1,1)
l2_init = l2_init.reshape(-1,1)
Lik_null = np.zeros((100,1))
Lik_full = np.zeros((100,1))


prior_lengthscale_da = tfp.distributions.Gamma(np.float64(alphaA), np.float64(betaA))
prior_lengthscale_db = tfp.distributions.Gamma(np.float64(alphaB), np.float64(betaB))
prior_variance_da = tfp.distributions.Gamma(np.float64(alpha_var), np.float64(beta_var))


for i in range(1,100):
    try:
        init_lengthscale_da = l1_init[i,0]
        init_lengthscale_db = l2_init[i,0]
        init_variance = (np.max(Effect_A)+np.max(Effect_B))/2
        init_likelihood_variance = 0.01

        k_full = K_multiplicative()
        m_full = gpflow.models.GPR(data=(Dose_AB, Effect), kernel=k_full,  mean_function=None)
        m_full.likelihood.variance.assign(0.01)
        #set_trainable(m_full.likelihood.variance, False)
        m_full.kernel.lengthscale_da.assign(init_lengthscale_da)
        m_full.kernel.lengthscale_db.assign(init_lengthscale_db)
        m_full.kernel.variance_da.assign(init_variance)
        # priors
        m_full.kernel.lengthscale_da.prior = prior_lengthscale_da
        m_full.kernel.lengthscale_db.prior = prior_lengthscale_db
        m_full.kernel.variance_da.prior = prior_variance_da
        m_full.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(2.0), np.float64(1.0))
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m_full.training_loss,
                                m_full.trainable_variables,method='BFGS',
                                options=dict(maxiter=100))
        #print_summary(m_full)
        Lik_full[i,0] = np.asarray(m_full.training_loss())
    except:
        Lik_full[i,0] = 'NaN'
        print('Cholesky was not successful')

index = np.where(Lik_full == np.nanmin(Lik_full))[0][0]
#index = np.where(Lik_full == np.nanmax(Lik_full))[0][0]

init_lengthscale_da = l1_init[index,0]
init_lengthscale_db = l2_init[index,0]

init_likelihood_variance = 0.01

k_full = K_multiplicative()
m_full = gpflow.models.GPR(data=(Dose_AB, Effect), kernel=k_full,  mean_function=None)
m_full.likelihood.variance.assign(init_likelihood_variance)
#set_trainable(m_full.likelihood.variance, False)
m_full.kernel.lengthscale_da.assign(init_lengthscale_da)
m_full.kernel.lengthscale_db.assign(init_lengthscale_db)
m_full.kernel.variance_da.assign(init_variance)
#priors
m_full.kernel.lengthscale_da.prior = prior_lengthscale_da
m_full.kernel.lengthscale_db.prior = prior_lengthscale_db
m_full.kernel.variance_da.prior = prior_variance_da
m_full.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(2.0), np.float64(1.0))
#m_null.likelihood.variance.trainable = False
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m_full.training_loss,
                    m_full.trainable_variables, method='BFGS',
                    options=dict(maxiter=1000))
print_summary(m_full)

# After optimization go back
Effect = df['Response'].values.reshape(-1,1).copy()
Dose_A = df['Dose1'].values.astype(float).copy()
Dose_B = df['Dose2'].values.astype(float).copy()
Dose_AB = np.concatenate((Dose_A.reshape(-1,1), Dose_B.reshape(-1,1)), axis=1)

Effect_B = df[df['Dose1'] == 0]['Response'].to_numpy().reshape(-1,1).astype(float)
Effect_A = df[df['Dose2'] == 0]['Response'].to_numpy().reshape(-1,1).astype(float)
Dose_A = df[df['Dose2']==0]['Dose1'].to_numpy().reshape(-1,1).astype(float)
Dose_B = df[df['Dose1']==0]['Dose2'].to_numpy().reshape(-1,1).astype(float)


num_predict = 2000
#[Xi, Xj] = np.meshgrid(X1, X2)
[Xi, Xj] = np.meshgrid(np.linspace(np.min(Dose_A), np.max(Dose_A), num_predict), np.linspace(np.min(Dose_B), np.max(Dose_B), num_predict))
X2 = Dose_AB.copy()

# We need to augument our test space to be a list of coordinates for input to the GP
Xnew2 = np.vstack((Xi.ravel(), Xj.ravel())).T # Change our input grid to list of coordinates

# Predict the mean and covariance of the GP fit at the test locations
mean2_full, Cov2_full = m_full.predict_f(Xnew2)

mean2_full = np.asarray(mean2_full)
Cov2_full = np.asarray(Cov2_full)



mean_full_est = pd.DataFrame(mean2_full.reshape(Xi.shape))
Cov_full_est = pd.DataFrame(Cov2_full.reshape(Xi.shape))

fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(Xi, Xj, mean2_full.reshape(Xi.shape), cmap=cm.viridis,
                       linewidth=0, antialiased=False)

Cov_full_est = pd.DataFrame(Cov2_full.reshape(Xi.shape))

plt.title("2d fit with GPs")
# Customize the z axis.
ax.set_zlim(0.01, 100.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('Dose A')
ax.set_ylabel('Dose B')
ax.set_zlabel('Effect')
ax.view_init(30, 70)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('figures/Greco/Greco'+'JointFit'+'.png')

xx_A = np.linspace(np.min(Dose_A), np.max(Dose_A),  num_predict).reshape(num_predict, 1)
xx_B = np.linspace(np.min(Dose_B), np.max(Dose_B),  num_predict).reshape( num_predict, 1)
## plot
plt.figure(figsize=(12, 6))
plt.plot(Dose_A, Effect_A, "kx", mew=2)
plt.plot(xx_A, mean_full_est.loc[0],"C0", lw=2, color='purple')
plt.fill_between(
    xx_A[:, 0],
    mean_full_est.loc[0] - 1.96 * np.sqrt(Cov_full_est.loc[0]),
    mean_full_est.loc[0] + 1.96 * np.sqrt( Cov_full_est.loc[0]),
    color="purple",
    alpha=0.2
)
#plt.title('Dose-response curve')
plt.ylabel('Response', fontsize=20)
plt.xlabel('$x_1$', fontsize=20)
plt.savefig('figures/Greco/Greco'+'DrugA'+'.png')

## plot
plt.figure(figsize=(12, 6))
plt.plot(np.log(Dose_A+m_full.kernel.lengthscale_da.value()), Effect_A, "kx", mew=2)
plt.plot(np.log(xx_A+m_full.kernel.lengthscale_da.value()), mean_full_est.loc[0],"C0", lw=2, color='purple')
plt.fill_between(
    np.log(xx_A[:, 0]+m_full.kernel.lengthscale_da.value()),
    mean_full_est.loc[0] - 1.96 * np.sqrt(Cov_full_est.loc[0]),
    mean_full_est.loc[0] + 1.96 * np.sqrt( Cov_full_est.loc[0]),
    color="purple",
    alpha=0.2
)
plt.ylabel('Response', fontsize=20)
plt.xlabel('$log(x_1+l_1)$', fontsize=20)
#plt.title('Dose-response curve')
plt.savefig('figures/Greco/Greco'+'DrugA_in_log'+'.png')

## plot
plt.figure(figsize=(12, 6))
plt.plot(np.log(Dose_B+m_full.kernel.lengthscale_db.value()), Effect_B, "kx", mew=2)
plt.plot(np.log(xx_B+m_full.kernel.lengthscale_db.value()), mean_full_est.iloc[:,0],"C0", lw=2, color='purple')
plt.fill_between(
    np.log(xx_B[:, 0]+m_full.kernel.lengthscale_db.value()),
    mean_full_est.iloc[:,0] - 1.96 * np.sqrt(Cov_full_est.iloc[:,0]),
    mean_full_est.iloc[:,0] + 1.96 * np.sqrt( Cov_full_est.iloc[:,0]),
    color="purple",
    alpha=0.2
)
plt.ylabel('Response', fontsize=20)
plt.xlabel('$log(x_2+l_2)$', fontsize=20)
plt.savefig('figures/Greco/Greco'+'DrugB_in_log'+'.png')

## plot
plt.figure(figsize=(12, 6))
plt.plot(Dose_B, Effect_B, "kx", mew=2)
plt.plot(xx_B, mean_full_est.iloc[:,0],"C0", lw=2, color='purple')
plt.fill_between(
    xx_B[:, 0],
    mean_full_est.iloc[:,0] - 1.96 * np.sqrt(Cov_full_est.iloc[:,0]),
    mean_full_est.iloc[:,0] + 1.96 * np.sqrt( Cov_full_est.iloc[:,0]),
    color="purple",
    alpha=0.2
)
plt.ylabel('Response', fontsize=20)
plt.xlabel('$x_2$', fontsize=20)
plt.savefig('figures/Greco/Greco'+'DrugB'+'.png')

dim2_A = mean_full_est.iloc[0].to_numpy()
dim2_B = mean_full_est.loc[:][0].to_numpy()

dim2_A_lower = mean_full_est.loc[0].to_numpy() - 1.96 * np.sqrt(Cov_full_est.loc[0]).to_numpy()
dim2_B_lower = mean_full_est.iloc[:,0].to_numpy() - 1.96 * np.sqrt(Cov_full_est.iloc[:,0]).to_numpy()

dim2_A_upper = mean_full_est.loc[0].to_numpy() + 1.96 * np.sqrt(Cov_full_est.loc[0]).to_numpy()
dim2_B_upper = mean_full_est.iloc[:,0].to_numpy() + 1.96 * np.sqrt(Cov_full_est.iloc[:,0]).to_numpy()

data = pd.concat([pd.DataFrame(Dose_AB), pd.DataFrame(Effect)], axis=1)
data.columns = ['Dose_A', 'Dose_B', 'Effect']

X1 = data[data['Dose_B']==0]['Dose_A'].to_numpy().reshape(-1,1).astype(float)
Y1 = data[data['Dose_B']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

X2 = data[data['Dose_A']==0]['Dose_B'].to_numpy().reshape(-1,1).astype(float)
Y2 = data[data['Dose_A']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

Y_expected_Hand = fit_Hand(X1, X2, dim2_A, dim2_B, Dose_A, Dose_B)

Y_expected_Hand_lower = fit_Hand(X1, X2, dim2_A_lower, dim2_B_lower, Dose_A, Dose_B)

Y_expected_Hand_upper = fit_Hand(X1, X2, dim2_A_upper, dim2_B_upper, Dose_A, Dose_B)

Y_expected_Hand_check = fit_Hand(X2, X1, dim2_B, dim2_A, Dose_B, Dose_A)

mean_full, Cov_full = predict_in_observations(X1, X2, m_full)

mean_full_lower, Cov_full_lower = predict_in_observations_lower(X1, X2, m_full)

mean_full_upper, Cov_full_upper = predict_in_observations_upper(X1, X2, m_full)


[Xi, Xj] = np.meshgrid(X1,X2)

xyz_full = np.concatenate((Xi.reshape(-1,1), Xj.reshape(-1,1), mean_full.reshape(-1,1)),axis=1)
xyz_full_lower = np.concatenate((Xi.reshape(-1,1), Xj.reshape(-1,1), mean_full_lower.reshape(-1,1)),axis=1)
xyz_full_upper = np.concatenate((Xi.reshape(-1,1), Xj.reshape(-1,1), mean_full_upper.reshape(-1,1)),axis=1)

Volume_full = trapezoidal_area(xyz_full)
Volume_full_lower = trapezoidal_area(xyz_full_lower)
Volume_full_upper = trapezoidal_area(xyz_full_upper)

xyz_null = np.concatenate((Xi.reshape(-1,1), Xj.reshape(-1,1), Y_expected_Hand.reshape(-1,1)),axis=1)
xyz_null_lower = np.concatenate((Xi.reshape(-1,1), Xj.reshape(-1,1), Y_expected_Hand_lower.reshape(-1,1)),axis=1)
xyz_null_upper = np.concatenate((Xi.reshape(-1,1), Xj.reshape(-1,1), Y_expected_Hand_upper.reshape(-1,1)),axis=1)

Volume_null = trapezoidal_area(xyz_null)
Volume_null_lower = trapezoidal_area(xyz_null_lower)
Volume_null_upper = trapezoidal_area(xyz_null_upper)

print('Volume difference', Volume_null-Volume_full)
print('Volume difference lower', Volume_null_lower-Volume_full_lower)
print('Volume difference upper', Volume_null_upper-Volume_full_upper)

xv, yv = np.meshgrid(X1, X2)
result = np.concatenate((xv.reshape(-1,1), yv.reshape(-1,1), (Y_expected_Hand - mean_full).reshape(-1,1)), axis = 1)
result_lower = np.concatenate((xv.reshape(-1,1), yv.reshape(-1,1), (Y_expected_Hand_lower - mean_full).reshape(-1,1)), axis = 1)
result_upper = np.concatenate((xv.reshape(-1,1), yv.reshape(-1,1), (Y_expected_Hand_upper - mean_full).reshape(-1,1)), axis = 1)

df = pd.DataFrame(result)
df.to_csv("results/Greco/results_Greco.csv")

df_lower = pd.DataFrame(result_lower)
df_lower.to_csv("results/Greco/results_Greco_lower.csv")

df_upper = pd.DataFrame(result_upper)
df_upper.to_csv("results/Greco/results_Greco_upper.csv")

exit()
xx_a = np.linspace(np.min(Dose_A), np.max(Dose_A), dim2_A.shape[0]).reshape(-1,1)
xx_b = np.linspace(np.min(Dose_B), np.max(Dose_B), dim2_B.shape[0]).reshape(-1,1)

fig, ax = plt.subplots(figsize=(6,6))
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
#ax.set_aspect('equal')
v = np.linspace(-10, 20, 11, endpoint=True)
cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(), Y_expected_Hand - mean_full,v,cmap='RdYlGn')
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)
#plt.title("Hand-GP estimated effect", fontsize=20)
plt.savefig('figures/Greco/Greco_contour_result'+'.png',bbox_inches = 'tight',
    pad_inches = 0)

fig, ax = plt.subplots(figsize=(6,6))
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
#ax.set_aspect('equal')
v = np.linspace(-10, 20, 11, endpoint=True)
cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(), Y_expected_Hand_upper - mean_full_upper,v,cmap='RdYlGn')
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)
#plt.title("Hand-GP estimated effect", fontsize=20)
plt.savefig('figures/Greco/Greco_contour_result_upper'+'.png',bbox_inches = 'tight',
    pad_inches = 0)

fig, ax = plt.subplots(figsize=(6,6))
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
#ax.set_aspect('equal')
v = np.linspace(-10, 20, 11, endpoint=True)
cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(), Y_expected_Hand_lower - mean_full_lower,v,cmap='RdYlGn')
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)
#plt.title("Hand-GP estimated effect", fontsize=20)
plt.savefig('figures/Greco/Greco_contour_result_lower'+'.png',bbox_inches = 'tight',
    pad_inches = 0)

fig, ax = plt.subplots(figsize=(6,6))
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
v = np.linspace(0.0, 120.0, 10, endpoint=True)
#ax.set_aspect('equal')
cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(),Y_expected_Hand, v)
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)
#plt.title("Hand-GP estimated surface", fontsize=20)
plt.savefig('figures/Greco/Greco_Hand_contour_result'+'.png',bbox_inches = 'tight',
    pad_inches = 0)

fig, ax = plt.subplots(figsize=(6,6))
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
#ax.set_aspect('equal')
v = np.linspace(0.0, 120.0, 10, endpoint=True)
cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(),mean_full, v)
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)

#plt.title("GP estimated surface", fontsize=20)
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)
plt.savefig('figures/Greco/Greco_mean_full_contour_result'+'.png',bbox_inches = 'tight',
    pad_inches = 0)

# avergage over multiple (0,0) points
df = pd.read_csv('../data/GrecoSimulatedData.csv', sep=';')
df = df.sort_values(by=['Dose1','Dose2'])

Effect = df['Response'].values.reshape(-1,1).copy()
df = df.iloc[2:]
df['Response'].iloc[0] = 106.7

Effect = df['Response'].values.reshape(-1,1).copy()
Dose_A = df['Dose1'].values.astype(float).copy()
Dose_B = df['Dose2'].values.astype(float).copy()
Dose_AB = np.concatenate((Dose_A.reshape(-1,1), Dose_B.reshape(-1,1)), axis=1)

Effect_B = df[df['Dose1'] == 0]['Response'].to_numpy().reshape(-1,1).astype(float)
Effect_A = df[df['Dose2'] == 0]['Response'].to_numpy().reshape(-1,1).astype(float)
Dose_A = df[df['Dose2']==0]['Dose1'].to_numpy().reshape(-1,1).astype(float)
Dose_B = df[df['Dose1']==0]['Dose2'].to_numpy().reshape(-1,1).astype(float)

data = pd.concat([pd.DataFrame(Dose_AB), pd.DataFrame(Effect)], axis=1)
data.columns = ['Dose_A', 'Dose_B', 'Effect']

X1 = data[data['Dose_B']==0]['Dose_A'].to_numpy().reshape(-1,1).astype(float)
Y1 = data[data['Dose_B']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

X2 = data[data['Dose_A']==0]['Dose_B'].to_numpy().reshape(-1,1).astype(float)
Y2 = data[data['Dose_A']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

mean_full, Cov_full = predict_in_observations(X1, X2, m_full)


fig, ax = plt.subplots(figsize=(6,6))
#ax.set_aspect('equal')
v = np.linspace(-8.0, 15.0, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(),mean_full-Effect.reshape(6,6).T,v,  cmap='YlGnBu')
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)
#plt.title("GP residuals", fontsize=20)
plt.savefig('figures/Greco/Greco_GP_residuals'+'.png',bbox_inches = 'tight',
    pad_inches = 0)

print('MSE total',np.sum((mean_full.flatten()-Effect.reshape(6,6).T.flatten())**2)/(len(mean_full.flatten())))

mean_full = pd.DataFrame(mean_full)
Cov_full = pd.DataFrame(Cov_full)

plt.figure(figsize=(12, 6))
plt.plot(Dose_A, Effect_A, "kx", mew=2)
plt.plot(X1.flatten(), mean_full.loc[0],"purple", lw=2, color='purple')
plt.fill_between(
    X1.flatten(),
    mean_full.loc[0] - 1.96 * np.sqrt(Cov_full.loc[0]),
    mean_full.loc[0] + 1.96 * np.sqrt( Cov_full.loc[0]),
    color="purple",
    alpha=0.2
)
#plt.title('Dose-response curve')
plt.ylabel('Response', fontsize=20)
plt.xlabel('$x_1$', fontsize=20)
plt.savefig('figures/Greco/Greco'+'DrugA'+'.png')

plt.figure(figsize=(12, 6))
plt.plot(Dose_B, Effect_B, "kx", mew=2)
plt.plot(X2.flatten(), mean_full.iloc[:,0],"purple", lw=2, color='purple')
plt.fill_between(
    X2.flatten(),
    mean_full.iloc[:,0] - 1.96 * np.sqrt(Cov_full.iloc[:,0]),
    mean_full.iloc[:,0] + 1.96 * np.sqrt( Cov_full.iloc[:,0]),
    color="purple",
    alpha=0.2
)
#plt.title('Dose-response curve')
plt.ylabel('Response', fontsize=20)
plt.xlabel('$x_1$', fontsize=20)
plt.savefig('figures/Greco/Greco'+'DrugB'+'.png')

df = pd.DataFrame(result)
#df.to_csv("results/Greco/results_Greco.csv")

'''
Run HMC.
General strategy is to run different combinations of the step size and leap frog
step and pick the largest step size with the highest acceptance probability.
'''

def plot_samples(step_size, num_leapfrog, samples, parameters, y_axis_label):
    # Plot samples of the pars
    num_parameters = 4
    figsize = (30, 20)
    fig, axes = plt.subplots(num_parameters, 1, figsize=figsize)
    axes[0].plot(samples[0])
    axes[0].tick_params(axis='x', labelsize=30)
    axes[0].tick_params(axis='y', labelsize=30)
    plt.ylabel(y_axis_label, fontsize=15)
    plt.xlabel("HMC iteration", fontsize=15)
    plt.ylabel(y_axis_label, fontsize=15)
    axes[1].plot(samples[1])
    axes[1].tick_params(axis='x', labelsize=30)
    axes[1].tick_params(axis='y', labelsize=30)
    plt.ylabel(y_axis_label, fontsize=15)
    plt.xlabel("HMC iteration", fontsize=15)
    plt.ylabel(y_axis_label, fontsize=15)
    axes[2].plot(samples[2])
    axes[2].tick_params(axis='x', labelsize=30)
    axes[2].tick_params(axis='y', labelsize=30)
    plt.ylabel(y_axis_label, fontsize=15)
    plt.xlabel("HMC iteration", fontsize=15)
    plt.ylabel(y_axis_label, fontsize=15)
    axes[3].plot(samples[3])
    # plt.figure(figsize=(8, 4))
    # for val, param in zip(samples, parameters):
    #     plt.plot(tf.squeeze(val), label=param_to_name[param])
    axes[3].tick_params(axis='x', labelsize=30)
    axes[3].tick_params(axis='y', labelsize=30)
    plt.legend(bbox_to_anchor=(1.0, 3.0))
    plt.xlabel("HMC iteration", fontsize=15)
    plt.ylabel(y_axis_label, fontsize=15)
    plt.savefig('figures/Greco/greco_samples'+str(step_size)+'_'+str(num_leapfrog)+'.png')


def marginal_samples(step_size,num_leapfrog, samples, parameters, y_axis_label):
    fig, axes = plt.subplots(1, len(param_to_name), figsize=(15, 3), constrained_layout=True)
    for ax, val, param in zip(axes, samples, parameters):
        ax.hist(np.stack(val).flatten(), bins=20)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_title(param_to_name[param])
    fig.suptitle(y_axis_label)
    plt.savefig('figures/Greco/greco_marginal_samples_'+str(step_size)+'_'+str(num_leapfrog)+'.png')

    res = np.zeros((4, 2))
    for i in range(4):
        hpd = pymc3.stats.hpd(np.asarray(samples[i]))
        res[i,0] = hpd[0]
        res[i,1] = hpd[1]
    print(res)

    res = np.concatenate((np.array(['l_a', 'l_b', 'var', 'var_lik']).reshape(-1,1), np.array([m_full.kernel.lengthscale_da.value(),
    m_full.kernel.lengthscale_db.value(), m_full.kernel.variance_da.value(), m_full.likelihood.variance.value()]).reshape(-1,1), res), axis=1)

    df = pd.DataFrame(res).round(2)
    print(df)

    df.columns = ['parameter','MAP','hpd_l', 'hpd_u']
    #df = df.round(2)
    #print( df.round(2))
    df['MAP'] = df['MAP'].astype(float).round(2)
    df['hpd_l'] = df['hpd_l'].astype(float).round(2)
    df['hpd_u'] = df['hpd_u'].astype(float).round(2)
    print(df)
    df.to_csv('results/Greco/greco_hyperparameters.csv')



parameters_vector = tf.stack(m_full.trainable_parameters)

num_burnin_steps = ci_niter(1000)
num_samples = ci_niter(100000)

num_leapfrog, step_size = [5, 0.7]

# Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
hmc_helper = gpflow.optimizers.SamplingHelper(
    m_full.log_posterior_density, m_full.trainable_parameters
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

param_to_name = {param: name for name, param in gpflow.utilities.parameter_dict(m_full).items()}

#marginal_samples(step_size,samples, m_full.trainable_parameters, "unconstrained variable samples")
marginal_samples(step_size,num_leapfrog,parameter_samples, m_full.trainable_parameters, "constrained parameter samples")

#plot_samples(samples, m_full.trainable_parameters, "unconstrained values")
plot_samples(step_size,num_leapfrog,parameter_samples, m_full.trainable_parameters, "constrained parameter values")
