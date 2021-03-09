import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability import mcmc
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
from gpflow import set_trainable
from scipy.interpolate import interp1d
from utilities import (compute_prior_hyperparameters, trapezoidal_area, fit_1d_model, predict_in_observations, fit_Hand, y_exp_Hand, y_exp, fit_3d_model, find_nearest, K_log, K_multiplicative)
import pymc3
f64 = gpflow.utilities.to_default_float


################################################################################################
# It is immportant to set up prior for all parameters, including the variance of the likelihood
################################################################################################

df = pd.read_csv('YonetaniData1.csv', sep=";")
#df = df.iloc[1:]
drug_name = 'Yonetani_'

print(df)


Effect = df['FracInhib'].values.reshape(-1,1).copy()
Dose_A = df['Conc.ADP'].values.astype(float).copy()
Dose_B = df['Conc.ADPr'].values.astype(float).copy()
Dose_AB = np.concatenate((Dose_A.reshape(-1,1), Dose_B.reshape(-1,1)),  axis=1)

Effect_B = df[df['Conc.ADP'] == 0]['FracInhib'].to_numpy().reshape(-1,1).astype(float)
Effect_A = df[df['Conc.ADPr'] == 0]['FracInhib'].to_numpy().reshape(-1,1).astype(float)
Dose_A = df[df['Conc.ADPr']==0]['Conc.ADP'].to_numpy().reshape(-1,1).astype(float)
Dose_B = df[df['Conc.ADP']==0]['Conc.ADPr'].to_numpy().reshape(-1,1).astype(float)


# hyperparameters of the priors
A_max  = np.max(Dose_A)
B_max = np.max(Dose_B)

alphaA, betaA = compute_prior_hyperparameters(30.0*A_max, 0.03*A_max)
alphaB, betaB = compute_prior_hyperparameters(B_max, 0.03*B_max)

eff_max_a = np.max(Effect_A)
eff_max_b = np.max(Effect_B)
eff_max = np.max([eff_max_a, eff_max_b])

print(eff_max)


alpha_var, beta_var = compute_prior_hyperparameters(eff_max, 0.0001*eff_max)



# # Fit univariate model
#
# k_univ = K_log()
#
#
# m = gpflow.models.GPR(data=(Dose_A, Effect_A), kernel=k_univ, mean_function=None)
#
# m.likelihood.variance.assign(0.01)
# set_trainable(m.likelihood.variance, False)
# init_lengthscale_da = alphaA
# m.kernel.lengthscale.assign(init_lengthscale_da)
# m.kernel.variance.assign(alpha_var/beta_var)
#
# m.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(2.0), np.float64(2.0))
# m.kernel.variance.prior = tfp.distributions.Gamma(np.float64(alpha_var), np.float64(beta_var))
#
#
# m.kernel.lengthscale.prior = tfp.distributions.Gamma(np.float64(alphaA), np.float64(betaA))
#
#
# opt = gpflow.optimizers.Scipy()
#
# opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
#
# print('Model for A')
# print_summary(m)
#
# #m.kernel.lengthscale.assign(0.77)
# #m.kernel.variance.assign(949.0)
#
# num_predict = 50
# xx_A = np.linspace(0.0, np.max(Dose_A),  num_predict).reshape(num_predict, 1)
# xx_B = np.linspace(0.0, np.max(Dose_B),  num_predict).reshape( num_predict, 1)
#
# mean2, Cov2 = m.predict_f(xx_A)
#
# #mean2 = pd.DataFrame(mean2)
# #Cov2 = pd.DataFrame(Cov2)
# ## plot
# plt.figure(figsize=(12, 6))
# plt.plot(Dose_A, Effect_A, "kx", mew=2)
# plt.plot(xx_A, mean2,"C0", lw=2, color='red')
# plt.fill_between(
#     xx_A[:, 0],
#     mean2[:, 0] - 1.96 * np.sqrt(Cov2[:,0]),
#     mean2[:, 0] + 1.96 * np.sqrt( Cov2[:,0]),
#     color="red",
#     alpha=0.2
# )
# plt.show()
#
#
# #Fit univariate model
#
# k_univ = K_log()
#
# m = gpflow.models.GPR(data=(Dose_B, Effect_B), kernel=k_univ, mean_function=None)
#
# m.likelihood.variance.assign(0.01)
# set_trainable(m.likelihood.variance, False)
# print('alphaB', alphaB/betaB)
#
# init_lengthscale_db = (alphaB/betaB)/3
# m.kernel.lengthscale.assign(init_lengthscale_db)
# m.kernel.variance.assign(alpha_var/beta_var)
#
# m.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(2.0), np.float64(2.0))
# m.kernel.variance.prior = tfp.distributions.Gamma(np.float64(alpha_var), np.float64(beta_var))
#
#
# m.kernel.lengthscale.prior = tfp.distributions.Gamma(np.float64(alphaB), np.float64(betaB))
#
# opt = gpflow.optimizers.Scipy()
#
# opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
#
# #m.kernel.lengthscale.assign(103.0)
# #m.kernel.variance.assign(949.0)
#
# print('Model for B')
# print_summary(m)
#
# num_predict = 50
# xx_A = np.linspace(0.0, np.max(Dose_A),  num_predict).reshape(num_predict, 1)
# xx_B = np.linspace(0.0, np.max(Dose_B),  num_predict).reshape( num_predict, 1)
#
# mean2, Cov2 = m.predict_f(xx_B)
#
# #mean2 = pd.DataFrame(mean2)
# #Cov2 = pd.DataFrame(Cov2)
# ## plot
# plt.figure(figsize=(12, 6))
# plt.plot(Dose_B, Effect_B, "kx", mew=2)
# plt.plot(xx_B, mean2,"C0", lw=2, color='red')
# plt.fill_between(
#     xx_B[:, 0],
#     mean2[:, 0] - 1.96 * np.sqrt(Cov2[:,0]),
#     mean2[:, 0] + 1.96 * np.sqrt( Cov2[:,0]),
#     color="red",
#     alpha=0.2
# )
# plt.show()
#
#
# #Fit univariate model
#
# k_univ = gpflow.kernels.RBF()
#
# m = gpflow.models.GPR(data=(Dose_B, Effect_B), kernel=k_univ, mean_function=None)
#
# m.likelihood.variance.assign(0.01)
# m.likelihood.variance.assign(0.01)
#
# set_trainable(m.likelihood.variance, False)
#
# print('alphaB', alphaB/betaB)
#
# init_lengthscale_db = (alphaB/betaB)
# m.kernel.lengthscales.assign(init_lengthscale_db)
# m.kernel.variance.assign(alpha_var/beta_var)
#
# m.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(2.0), np.float64(2.0))
# m.kernel.variance.prior = tfp.distributions.Gamma(np.float64(alpha_var), np.float64(beta_var))
#
#
# m.kernel.lengthscales.prior = tfp.distributions.Gamma(np.float64(alphaB), np.float64(betaB))
#
# opt = gpflow.optimizers.Scipy()
#
# opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
#
# #m.kernel.lengthscale.assign(103.0)
# #m.kernel.variance.assign(949.0)
#
# print('Model for B')
# print_summary(m)
#
# num_predict = 50
# xx_A = np.linspace(0.0, np.max(Dose_A),  num_predict).reshape(num_predict, 1)
# xx_B = np.linspace(0.0, np.max(Dose_B),  num_predict).reshape( num_predict, 1)
#
# mean2, Cov2 = m.predict_y(xx_B)
#
# #mean2 = pd.DataFrame(mean2)
# #Cov2 = pd.DataFrame(Cov2)
# ## plot
# plt.figure(figsize=(12, 6))
# plt.plot(Dose_B, Effect_B, "kx", mew=2)
# plt.plot(xx_B, mean2,"C0", lw=2, color='red')
# plt.fill_between(
#     xx_B[:, 0],
#     mean2[:, 0] - 1.96 * np.sqrt(Cov2[:,0])- (1.96 * 37),
#     mean2[:, 0] + 1.96 * np.sqrt( Cov2[:,0])+ (1.96 * 37) ,
#     color="red",
#     alpha=0.2
# )
# plt.fill_between(
#     xx_B[:, 0],
#     mean2[:, 0] - 1.96 * np.sqrt(Cov2[:,0]),
#     mean2[:, 0] + 1.96 * np.sqrt( Cov2[:,0]),
#     color="red",
#     alpha=0.2
# )
# plt.show()
#

zeros_A = np.zeros((Dose_A.shape))
zeros_B = np.zeros((Dose_B.shape))

Dose_A_mono = np.concatenate((Dose_A.reshape(-1,1), zeros_A.reshape(-1,1)),  axis=0)
Dose_B_mono = np.concatenate((zeros_B.reshape(-1,1), Dose_B.reshape(-1,1)),  axis=0)

Dose_AB_mono = np.concatenate((Dose_A_mono.reshape(-1,1), Dose_B_mono.reshape(-1,1)),  axis=1)
Effect_mono = np.concatenate((Effect_A.reshape(-1,1), Effect_B.reshape(-1,1)),  axis=0)
#
# Dose_AB = np.concatenate((Dose_AB, Dose_AB_mono, Dose_AB_mono, Dose_AB_mono, Dose_AB_mono),  axis=0)
# Effect = np.concatenate((Effect.reshape(-1,1), Effect_mono.reshape(-1,1), Effect_mono.reshape(-1,1), Effect_mono.reshape(-1,1), Effect_mono.reshape(-1,1)),  axis=0)

#Dose_AB = np.concatenate(( Dose_AB_mono, Dose_AB_mono, Dose_AB_mono, Dose_AB_mono),  axis=0)
#Effect = np.concatenate(( Effect_mono.reshape(-1,1), Effect_mono.reshape(-1,1), Effect_mono.reshape(-1,1), Effect_mono.reshape(-1,1)),  axis=0)

Dose_AB = np.concatenate((Dose_AB, Dose_AB_mono, Dose_AB_mono, Dose_AB_mono, Dose_AB_mono),  axis=0)
Effect = np.concatenate((Effect.reshape(-1,1), Effect_mono.reshape(-1,1), Effect_mono.reshape(-1,1), Effect_mono.reshape(-1,1), Effect_mono.reshape(-1,1)),  axis=0)

#data_save = pd.concat([pd.DataFrame(Dose_AB), pd.DataFrame(Effect)], axis=1)
#data_save.columns = ['Dose_A', 'Dose_B', 'Effect']
#data_save.to_csv('Yonetani1_weighted.csv')

#exit()

#print('Dose_AB mono', Dose_AB)
#print('Effect mono', Effect)


#[l1_init, l2_init] = np.meshgrid(np.linspace(np.max(Dose_AB[:,0])/10.0, np.max(Dose_AB[:,0]), 10), np.linspace(np.max(Dose_AB[:,1])/10.0,  np.max(Dose_AB[:,1]), 10))
[l1_init, l2_init] = np.meshgrid(np.linspace(0.1, np.max(Dose_A), 10), np.linspace(10.0,  np.max(Dose_B), 10))
print('l1_init', l1_init)
print('l2_init', l2_init)
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


        #k1 = gpflow.kernels.RBF(active_dims=[0])
        #k2 = gpflow.kernels.RBF(active_dims=[1])
        #k3 = gpflow.kernels.RBF(active_dims=[2])
        #k = k1 * k2 #* k3
        k = K_multiplicative()

        m = gpflow.models.GPR(data=(Dose_AB, Effect), kernel=k, mean_function=None)

        m.likelihood.variance.assign(0.01)
        set_trainable(m.likelihood.variance, False)

        m.kernel.lengthscale_da.assign(init_lengthscale_da)
        m.kernel.lengthscale_db.assign(init_lengthscale_db)
        m.kernel.variance_da.assign(eff_max)

        m.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(2.0), np.float64(2.0))
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


#index = np.where(Lik_full == np.nanmax(Lik_full))[0][0]
index = np.where(Lik_full == np.nanmin(Lik_full))[0][0]

init_lengthscale_da = l1_init[index,0]
init_lengthscale_db = l2_init[index,0]
init_var = var_init[index,0]

print('init_lengthscale_da', init_lengthscale_da)
print('init_lengthscale_db', init_lengthscale_db)
print('init_lengthscale_db', init_var)

m = gpflow.models.GPR(data=(Dose_AB, Effect), kernel=k, mean_function=None)

m.likelihood.variance.assign(0.01)
set_trainable(m.likelihood.variance, False)


m.kernel.lengthscale_da.assign(init_lengthscale_da)
m.kernel.lengthscale_db.assign(init_lengthscale_db)
m.kernel.variance_da.assign(init_var)
# priors
m.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(2.0), np.float64(2.0))
m.kernel.variance_da.prior = tfp.distributions.Gamma(np.float64(alpha_var), np.float64(beta_var))

m.kernel.lengthscale_da.prior = tfp.distributions.Gamma(np.float64(alphaA), np.float64(betaA))
m.kernel.lengthscale_db.prior = tfp.distributions.Gamma(np.float64(alphaB), np.float64(betaB))


opt = gpflow.optimizers.Scipy()

opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

#m.kernel.lengthscale_da.assign(0.1)

print_summary(m)

[Xi, Xj] = np.meshgrid(np.linspace(0.0, np.max(Dose_A), 50), np.linspace(0.0, np.max(Dose_B), 50))
#X2 = Dose_A_t.copy()

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
#plt.ylim((0.0, 1.0))
#plt.title('Monotherapeutic slice of the GP surface', fontsize=20)
#plt.xlabel("$Dose$", fontsize=60), plt.ylabel("Effect", fontsize=60)
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('Response', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.savefig(drug_name+'_DrugA_2drugs'+'.png')

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
#plt.title('Monotherapeutic slice of the GP surface', fontsize=20)
#plt.xlabel("$Dose$", fontsize=60), plt.ylabel("Effect", fontsize=60)
plt.xlabel('$x_2$', fontsize=20)
plt.ylabel('Response', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.savefig(drug_name+'_DrugB_2drugs'+'.png')


dim2_A = mean2.iloc[0].to_numpy()
dim2_B = mean2.loc[:][0].to_numpy()

N = 5000

Effect = df['FracInhib'].values.reshape(-1,1).copy()
Dose_A = df['Conc.ADP'].values.astype(float).copy()
Dose_B = df['Conc.ADPr'].values.astype(float).copy()
Dose_AB = np.concatenate((Dose_A.reshape(-1,1), Dose_B.reshape(-1,1)),  axis=1)

Effect_B = df[df['Conc.ADP'] == 0]['FracInhib'].to_numpy().reshape(-1,1).astype(float)
Effect_A = df[df['Conc.ADPr'] == 0]['FracInhib'].to_numpy().reshape(-1,1).astype(float)
Dose_A = df[df['Conc.ADPr']==0]['Conc.ADP'].to_numpy().reshape(-1,1).astype(float)
Dose_B = df[df['Conc.ADP']==0]['Conc.ADPr'].to_numpy().reshape(-1,1).astype(float)

data = pd.concat([pd.DataFrame(Dose_AB), pd.DataFrame(Effect)], axis=1)
data.columns = ['Dose_A', 'Dose_B', 'Effect']

X1 = data[data['Dose_B']==0]['Dose_A'].to_numpy().reshape(-1,1).astype(float)
Y1 = data[data['Dose_B']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

X2 = data[data['Dose_A']==0]['Dose_B'].to_numpy().reshape(-1,1).astype(float)
Y2 = data[data['Dose_A']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

Y_expected_Hand = fit_Hand(N, X1, X2, dim2_A, dim2_B, Dose_A, Dose_B)
Y_expected_Hand_check = fit_Hand(N, X2, X1, dim2_B, dim2_A, Dose_B, Dose_A)

mean_full, Cov_full = predict_in_observations(X1, X2, m)

print(mean_full.shape)

[Xi, Xj] = np.meshgrid(X1,X2)

xyz_full = np.concatenate((Xi.reshape(-1,1), Xj.reshape(-1,1), mean_full.reshape(-1,1)),axis=1)
Volume_full = trapezoidal_area(xyz_full)


xyz_null = np.concatenate((Xi.reshape(-1,1), Xj.reshape(-1,1), Y_expected_Hand.reshape(-1,1)),axis=1)
Volume_null = trapezoidal_area(xyz_null)


print(Volume_null-Volume_full)



xv, yv = np.meshgrid(X1, X2)
#result = np.concatenate((xv.reshape(-1,1), yv.reshape(-1,1), (Y_expected_Hand - mean_full).reshape(-1,1)), axis = 1)

print(Y_expected_Hand - Y_expected_Hand_check)

xx_a = np.linspace(np.min(Dose_A), np.max(Dose_A), dim2_A.shape[0]).reshape(-1,1)
xx_b = np.linspace(np.min(Dose_B), np.max(Dose_B), dim2_B.shape[0]).reshape(-1,1)

#v = np.linspace(0.0, 1.0, 10, endpoint=True)

print('Dose_A', Dose_A, Dose_A.shape)
print('Dose_B', Dose_B, Dose_B.shape)
print('Y_expected_Hand', Y_expected_Hand.shape)

# plt.figure(figsize=(12, 6))
# plt.contourf(np.log(Dose_A.flatten()+np.array(m.kernel.lengthscale_da.value())), np.log(Dose_B.flatten()+np.array(m.kernel.lengthscale_db.value())),Y_expected_Hand)
# plt.colorbar()
# plt.xlabel('log($x_1$ +$l_a$)', fontsize=30)
# plt.ylabel('log($x_2$ +$l_b$)', fontsize=30)
# plt.title('Hand model contour', fontsize=30)
# plt.savefig(str(drug_name)+'_Hand_contour_result'+'.png')

fig, ax = plt.subplots(figsize=(6,6))
v = np.linspace(150.0, 1050.0, 10, endpoint=True)
#ax.set_aspect('equal')
#cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(),Y_expected_Hand,v)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(np.log(Dose_A.flatten()/np.array(m.kernel.lengthscale_da.value())+1), np.log(Dose_B.flatten()/np.array(m.kernel.lengthscale_db.value())+1),Y_expected_Hand,v)
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)

plt.xlabel('log($x_1$/$l_1$+1)', fontsize=15)
plt.ylabel('log($x_2$/$l_2$+1)', fontsize=15)
#plt.title("Hand-GP estimated surface", fontsize=20)
plt.savefig(str(drug_name)+'_Hand_contour_result'+'.png', bbox_inches = 'tight',
    pad_inches = 0)
# plt.figure(figsize=(12, 6))
# plt.contourf(np.log(Dose_A.flatten()+np.array(m.kernel.lengthscale_da.value())), np.log(Dose_B.flatten()+np.array(m.kernel.lengthscale_db.value())), mean_full)
# plt.colorbar()
# plt.xlabel('$x_1$ ', fontsize=30)
# plt.ylabel('$x_2$', fontsize=30)
# plt.title('GP mean contour', fontsize=30)
# plt.savefig(str(drug_name)+'_GP_mean'+'.png')

fig, ax = plt.subplots(figsize=(6,6))
#ax.set_aspect('equal')
v = np.linspace(0.0, 120.0, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
#cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(),mean_full)
cf = ax.contourf(np.log(Dose_A.flatten()/np.array(m.kernel.lengthscale_da.value())+1), np.log(Dose_B.flatten()/np.array(m.kernel.lengthscale_db.value())+1),mean_full)
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)

plt.xlabel('log($x_1$/$l_1$+1)', fontsize=15)
plt.ylabel('log($x_2$/$l_2$+1)', fontsize=15)
#plt.title("GP estimated surface", fontsize=20)
plt.savefig(str(drug_name)+'_GP_mean'+'.png', bbox_inches = 'tight',
    pad_inches = 0)


# #v = np.linspace(-1.0, 1.0, 10, endpoint=True)
# plt.figure(figsize=(12, 6))
# v = np.linspace(-1.0, 1.0, 10, endpoint=True)
# plt.contourf(np.log(Dose_A.flatten()+np.array(m.kernel.lengthscale_da.value())),
# np.log(Dose_B.flatten()+np.array(m.kernel.lengthscale_db.value())), Y_expected_Hand - mean_full, cmap='RdYlGn')
# plt.colorbar()
# plt.xlabel('log($x_1$ + $l_a$)', fontsize=30)
# plt.ylabel('log($x_2$ + $l_b$)', fontsize=30)
# plt.title('Difference between GP and Hand-GP', fontsize=30)
# plt.savefig(str(drug_name)+'contour_result'+'.png')

fig, ax = plt.subplots(figsize=(6,6))
#ax.set_aspect('equal')
v = np.linspace(-80.0, 240.0, 20, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
#cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(),Y_expected_Hand - mean_full,cmap='RdYlGn')
cf = ax.contourf(np.log(Dose_A.flatten()/np.array(m.kernel.lengthscale_da.value())+1), np.log(Dose_B.flatten()/np.array(m.kernel.lengthscale_db.value())+1),Y_expected_Hand - mean_full,v,cmap='RdYlGn')
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)

plt.xlabel('log($x_1$/$l_1$+1)', fontsize=15)
plt.ylabel('log($x_2$/$l_2$+1)', fontsize=15)
#plt.title("Hand-GP estimated effect", fontsize=20)
plt.savefig(str(drug_name)+'contour_result'+'.png', bbox_inches = 'tight',
    pad_inches = 0)



fig, ax = plt.subplots(figsize=(6,6))
#ax.set_aspect('equal')
v = np.linspace(-24.0,12.0, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
#cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(),mean_full - Effect.reshape(6,6), v, cmap='YlGnBu')
cf = ax.contourf(np.log(Dose_A.flatten()/np.array(m.kernel.lengthscale_da.value())+1), np.log(Dose_B.flatten()/np.array(m.kernel.lengthscale_db.value())+1),mean_full - Effect.reshape(6,6), v, cmap='YlGnBu')

cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)

plt.xlabel('log($x_1$/$l_1$+1)', fontsize=15)
plt.ylabel('log($x_2$/$l_2$+1)', fontsize=15)
#plt.title("GP residuals", fontsize=20)
plt.savefig(str(drug_name)+'_GP_residuals'+'.png')

data = {'drug1.conc':  [0.0,0.5, 1.0, 1.5, 2.0, 2.5],
        'drug2.conc': [0.0,95.0, 190.0, 285.0, 380.0, 475.0],
         'effect': [1000.0,528.0, 363.0, 266.0, 219.0, 183.0]
        }

df_diagonal = pd.DataFrame (data, columns = ['drug1.conc','drug2.conc','effect'])

x = np.linspace(0.0,2.5,100)
y = 190.0 * x

#[Xi, Xj] = np.meshgrid(x, 50), np.linspace(y, 50))

#X2 = Dose_A_t.copy()

# We need to augument our test space to be a list of coordinates for input to the GP
Xnew2 = np.vstack((x, y)).T # Change our input grid to list of coordinates

# Predict the mean and covariance of the GP fit at the test locations
mean2, Cov2 = m.predict_f(Xnew2)
print(mean2)

plt.figure(figsize=(12, 6))
plt.plot(np.asarray(np.sqrt(x**2+y**2)).flatten(), np.asarray(mean2).flatten(), "r", mew=2)
plt.plot(np.sqrt(df_diagonal['drug1.conc']**2+df_diagonal['drug2.conc']**2), df_diagonal['effect'], "kx", lw=2)
#plt.ylim((0.0, 1.0))
#plt.title('Diagonal slice of the GP surface', fontsize=20)
plt.xlabel('$x_1$+$x_2$', fontsize=20)
plt.ylabel('Response', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('ChouTalalay_GP'+'_diagonal'+'.png')

#Y_expected_Hand = fit_Hand(N, x, y, dim2_A, dim2_B, Dose_A, Dose_B)

#power_smooth = interp1d(np.sqrt(X1**2+X2**2), np.asarray(np.diag(Y_expected_Hand)).flatten(), xnew, kind='nearest')

f1 = interp1d(np.sqrt(X1**2+X2**2).flatten(), np.asarray(np.diag(Y_expected_Hand)).flatten(), kind='quadratic')

#xnew = np.linspace(np.sqrt(X1**2+X2**2).flatten().min(), np.sqrt(X1**2+X2**2).flatten().max(), 100)
xnew = np.asarray(np.sqrt(df_diagonal['drug1.conc']**2+df_diagonal['drug2.conc']**2)).flatten()
xnew = np.asarray(np.sqrt(Xnew2[:,0]**2+Xnew2[:,1]**2)).flatten()

#Xnew2 = np.vstack((np.asarray(df_diagonal['drug1.conc']), np.asarray(df_diagonal['drug2.conc']))).T # Change our input grid to list of coordinates
## Predict the mean and covariance of the GP fit at the test locations
#mean2, Cov2 = m.predict_f(Xnew2)
#print(mean2)


plt.figure(figsize=(12, 6))
plt.plot(np.asarray(np.sqrt(Xnew2[:,0]**2+Xnew2[:,1]**2)).flatten(), np.asarray(mean2).flatten(),  "purple", mew=2,label='Unconstrained')
#plt.plot(np.asarray(np.sqrt(X1**2+X2**2)).flatten(), np.asarray(np.diag(Y_expected_Hand)).flatten(), "grey", mew=2)
plt.plot(xnew, f1(xnew), "grey", mew=2, label='Null')
plt.plot(np.sqrt(df_diagonal['drug1.conc']**2+df_diagonal['drug2.conc']**2), df_diagonal['effect'], "kx", lw=2)
#plt.ylim((0.0, 1.0))
#plt.title('Diagonal slice of the GP surface', fontsize=20)
plt.xlabel('Diagonal combination of the doses', fontsize=20)
plt.ylabel('Response', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend( prop={'size': 20})
plt.savefig('ChouTalalay_GP'+'_diagonal_null'+'.png', bbox_inches = 'tight',
    pad_inches = 0)


diagonal_difference =  f1(xnew) - np.asarray(mean2).flatten()

length_diagonal = len(diagonal_difference)
xnew = (xnew - xnew.min())/(xnew.max()-xnew.min())

plt.figure(figsize=(12, 6))
plt.plot(xnew[0:9], diagonal_difference[0:9], "red", mew=2)
plt.plot(xnew[8:length_diagonal], diagonal_difference[8:length_diagonal], "green", mew=2)

#plt.plot(np.asarray(np.sqrt(X1**2+X2**2)).flatten(), np.asarray(np.diag(Y_expected_Hand)).flatten(), "grey", mew=2)
#plt.plot(xnew, f1(xnew), "grey", mew=2)
#plt.plot(np.sqrt(df_diagonal['drug1.conc']**2+df_diagonal['drug2.conc']**2), df_diagonal['effect'], "kx", lw=2)
#plt.ylim((0.0, 1.0))
#plt.title('Diagonal slice of the GP surface', fontsize=20)
plt.xlabel('Diagonal combination of the doses normalized', fontsize=20)
plt.ylabel('Response', fontsize=20)
plt.ylim(-50,200)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('ChouTalalay_GP'+'_diagonal_null_difference'+'.png', bbox_inches = 'tight',
    pad_inches = 0)

# #HMC
#
# num_burnin_steps = ci_niter(1000)
# num_samples = ci_niter(10000)
#
# # Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
# hmc_helper = gpflow.optimizers.SamplingHelper(
#     m.log_posterior_density, m.trainable_parameters
# )
#
# hmc = tfp.mcmc.HamiltonianMonteCarlo(
#     target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=5, step_size=0.03
# )
# adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
#     hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
# )
#
#
# @tf.function
# def run_chain_fn():
#     return tfp.mcmc.sample_chain(
#         num_results=num_samples,
#         num_burnin_steps=num_burnin_steps,
#         current_state=hmc_helper.current_state,
#         kernel=adaptive_hmc,
#         trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
#     )
#
#
# samples, traces = run_chain_fn()
# parameter_samples = hmc_helper.convert_to_constrained_values(samples)
#
# param_to_name = {param: name for name, param in gpflow.utilities.parameter_dict(m).items()}
#
#
# def plot_samples(samples, parameters, y_axis_label):
#     plt.figure(figsize=(8, 4))
#     for val, param in zip(samples, parameters):
#         plt.plot(tf.squeeze(val), label=param_to_name[param])
#     plt.legend(bbox_to_anchor=(1.0, 3.0))
#     plt.xlabel("HMC iteration")
#     plt.ylabel(y_axis_label)
#     plt.savefig('LatTer_samples.png')
#
#
# plot_samples(samples, m.trainable_parameters, "unconstrained values")
# plot_samples(parameter_samples, m.trainable_parameters, "constrained parameter values")
#
# def marginal_samples(samples, parameters, y_axis_label):
#     fig, axes = plt.subplots(1, len(param_to_name), figsize=(15, 3), constrained_layout=True)
#     for ax, val, param in zip(axes, samples, parameters):
#         ax.hist(np.stack(val).flatten(), bins=20)
#         ax.set_title(param_to_name[param])
#     fig.suptitle(y_axis_label)
#     plt.savefig('LatTer_marginal_samples.png')
#     plt.show()
#
#
#
# marginal_samples(samples, m.trainable_parameters, "unconstrained variable samples")
# marginal_samples(parameter_samples, m.trainable_parameters, "constrained parameter samples")


#HMC
#def plot_samples(step_size, num_leapfrog, samples, parameters, y_axis_label):
#    plt.figure(figsize=(8, 4))
#    for val, param in zip(samples, parameters):
#        plt.plot(tf.squeeze(val), label=param_to_name[param])
#    plt.legend(bbox_to_anchor=(1.0, 3.0))
#    plt.xlabel("HMC iteration")
#    plt.ylabel(y_axis_label)
#    plt.savefig('ChouTalalay_samples'+str(step_size)+str(num_leapfrog)+'.png')

# def plot_samples(step_size, num_leapfrog, samples, parameters, y_axis_label):
#     fig, axes = plt.subplots(1, len(param_to_name), figsize=(15, 3), constrained_layout=True)
#     for ax, val, param in zip(axes, samples, parameters):
#         ax.plot(np.stack(val).flatten())
#         ax.set_title(param_to_name[param])
#     fig.suptitle(y_axis_label)
#     plt.savefig('ChouTalalay_samples'+str(step_size)+str(num_leapfrog)+'.png')
#
# def marginal_samples(step_size,leap_frog, samples, parameters, y_axis_label):
#     fig, axes = plt.subplots(1, len(param_to_name), figsize=(15, 3), constrained_layout=True)
#     for ax, val, param in zip(axes, samples, parameters):
#         ax.hist(np.stack(val).flatten(), bins=20)
#         ax.set_title(param_to_name[param])
#     fig.suptitle(y_axis_label)
#     plt.savefig('ChouTalalay_marginal_samples_'+str(step_size)+str(num_leapfrog)+'.png')
#     #plt.show()
#     res = np.zeros((4, 2))
#     for i in range(3):
#         hpd = pymc3.stats.hpd(np.asarray(samples[i]))
#         res[i,0] = hpd[0]
#         res[i,1] = hpd[1]
#     print(res)
#     res = np.concatenate((np.array(['l_a', 'l_b', 'var', 'var_lik']).reshape(-1,1), np.array([m.kernel.lengthscale_da.value(),
#     m.kernel.lengthscale_db.value(), m.kernel.variance_da.value(), m.likelihood.variance.value()]).reshape(-1,1), res), axis=1)
#
#     df = pd.DataFrame(res)
#     print(df)
#
#     df.columns = ['parameter','MAP','hpd_l', 'hpd_u']
#
#     print(df)
#     df = df.round(5)
#     df.to_csv('ChouTalalay_hyperparameters'+str(step_size)+str(leap_frog)+'.csv')
#
# leapfrog_num_and_step_size = np.stack(np.meshgrid([5.0], [0.02])).T.reshape(-1, 2)
# num_hmc_parameters = leapfrog_num_and_step_size.shape[0]
#
# for j in range(num_hmc_parameters):
#     try:
#         num_burnin_steps = ci_niter(1000)
#         num_samples = ci_niter(100000)
#
#         num_leapfrog, step_size = leapfrog_num_and_step_size[j]
#
#         # Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
#         hmc_helper = gpflow.optimizers.SamplingHelper(
#             m.log_posterior_density, m.trainable_parameters
#         )
#
#         hmc = tfp.mcmc.HamiltonianMonteCarlo(
#             target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=num_leapfrog, step_size=step_size
#         )
#         adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
#             hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
#         )
#
#
#         @tf.function
#         def run_chain_fn():
#             return tfp.mcmc.sample_chain(
#                 num_results=num_samples,
#                 num_burnin_steps=num_burnin_steps,
#                 current_state=hmc_helper.current_state,
#                 kernel=adaptive_hmc,
#                 trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
#             )
#
#
#         samples, traces = run_chain_fn()
#         parameter_samples = hmc_helper.convert_to_constrained_values(samples)
#
#         param_to_name = {param: name for name, param in gpflow.utilities.parameter_dict(m).items()}
#
#         #marginal_samples(step_size,samples, m_full.trainable_parameters, "unconstrained variable samples")
#         marginal_samples(step_size,num_leapfrog,parameter_samples, m.trainable_parameters, "constrained parameter samples")
#
#         #plot_samples(samples, m_full.trainable_parameters, "unconstrained values")
#         plot_samples(step_size,num_leapfrog,parameter_samples, m.trainable_parameters, "constrained parameter values")
#
#     except:
#         print('HMC did not work')
