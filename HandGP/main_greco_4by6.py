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
import scipy

from utilities import (trapezoidal_area, compute_prior_hyperparameters,  predict_in_observations, predict_in_observations_lower,
                       predict_in_observations_upper, fit_Hand, y_exp_Hand, y_exp, find_nearest, K_log, K_multiplicative)

f64 = gpflow.utilities.to_default_float

df = pd.read_csv('../data/main_greco_3times6.csv', sep=';')

print(df)

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

c_a = eff_max_a/eff_min_a
c_b = eff_max_b/eff_min_b

alphaA, betaA = compute_prior_hyperparameters(A_max/2, 0.1*A_max/c_a)
alphaB, betaB = compute_prior_hyperparameters(B_max/2, 0.1*B_max/c_b)

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

Dose_A_mono = np.concatenate((Dose_A.reshape(-1,1), zeros_B.reshape(-1,1)),  axis=0)
Dose_B_mono = np.concatenate((zeros_A.reshape(-1,1), Dose_B.reshape(-1,1)),  axis=0)

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
        Lik_full[i,0] = np.asarray(m_full.training_loss())
    except:
        Lik_full[i,0] = 'NaN'
        print('Cholesky was not successful')

index = np.where(Lik_full == np.nanmin(Lik_full))[0][0]

init_lengthscale_da = l1_init[index,0]
init_lengthscale_db = l2_init[index,0]

init_likelihood_variance = 0.01

k_full = K_multiplicative()
m_full = gpflow.models.GPR(data=(Dose_AB, Effect), kernel=k_full,  mean_function=None)
m_full.likelihood.variance.assign(init_likelihood_variance)
m_full.kernel.lengthscale_da.assign(init_lengthscale_da)
m_full.kernel.lengthscale_db.assign(init_lengthscale_db)
m_full.kernel.variance_da.assign(init_variance)
#priors
m_full.kernel.lengthscale_da.prior = prior_lengthscale_da
m_full.kernel.lengthscale_db.prior = prior_lengthscale_db
m_full.kernel.variance_da.prior = prior_variance_da
m_full.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(2.0), np.float64(1.0))
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m_full.training_loss,
                    m_full.trainable_variables, method='BFGS',
                    options=dict(maxiter=1000))

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
plt.savefig('figures/Greco_4by6/Greco'+'JointFit'+'.png')

xx_A = np.linspace(np.min(Dose_A), np.max(Dose_A),  num_predict).reshape(num_predict, 1)
xx_B = np.linspace(np.min(Dose_B), np.max(Dose_B),  num_predict).reshape(num_predict, 1)
## plot
plt.figure(figsize=(12, 10))
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
plt.ylabel('Response', fontsize=40)
plt.xlabel('$x_1$', fontsize=40)
plt.tick_params(axis='both', which='major', labelsize=40)
plt.savefig('figures/Greco_4by6/Greco'+'DrugA'+'.png', bbox_inches="tight")

## plot
plt.figure(figsize=(12, 10))
plt.plot(Dose_B, Effect_B, "kx", mew=2)
plt.plot(xx_B, mean_full_est.iloc[:,0],"C0", lw=2, color='purple')
plt.fill_between(
    xx_B[:, 0],
    mean_full_est.iloc[:,0] - 1.96 * np.sqrt(Cov_full_est.iloc[:,0]),
    mean_full_est.iloc[:,0] + 1.96 * np.sqrt( Cov_full_est.iloc[:,0]),
    color="purple",
    alpha=0.2
)
plt.ylabel('Response', fontsize=40)
plt.xlabel('$x_2$', fontsize=40)
plt.tick_params(axis='both', which='major', labelsize=40)
plt.savefig('figures/Greco_4by6/Greco'+'DrugB'+'.png', bbox_inches="tight")

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

mean_full, Cov_full = predict_in_observations(X1, X2, m_full)

mean_full, Cov_full = predict_in_observations(X1, X2, m_full)

mean_full_lower, Cov_full_lower = predict_in_observations_lower(X1, X2, m_full)

mean_full_upper, Cov_full_upper = predict_in_observations_upper(X1, X2, m_full)

[Xi, Xj] = np.meshgrid(X1,X2)

[Xi, Xj] = np.meshgrid(X1,X2)

xv, yv = np.meshgrid(X1, X2)

xx_a = np.linspace(np.min(Dose_A), np.max(Dose_A), dim2_A.shape[0]).reshape(-1,1)
xx_b = np.linspace(np.min(Dose_B), np.max(Dose_B), dim2_B.shape[0]).reshape(-1,1)

fig, ax = plt.subplots(figsize=(6,6))
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
#ax.set_aspect('equal')
v = np.linspace(-60, 30, 11, endpoint=True)
cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(), Y_expected_Hand - mean_full,v,cmap='RdYlGn')
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)
#plt.title("Hand-GP estimated effect", fontsize=20)
plt.savefig('figures/Greco_4by6/Greco_contour_result'+'.png',bbox_inches = 'tight',
    pad_inches = 0)
