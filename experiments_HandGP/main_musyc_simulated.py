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
from gpflow import set_trainable
from utilities import (compute_prior_hyperparameters, trapezoidal_area, fit_1d_model, predict_in_observations, fit_Hand, y_exp_Hand, y_exp, fit_3d_model, find_nearest, K_log, K_multiplicative)

f64 = gpflow.utilities.to_default_float


################################################################################################
# It is immportant to set up prior for all parameters, including the variance of the likelihood
################################################################################################

df = pd.read_csv('musyc_simulated.csv', sep=",")
#df = df.iloc[1:]
drug_name = 'musyc_'

print(df)

Effect = df['effect'].values.reshape(-1,1).copy()
Dose_A = df['drug1.conc'].values.astype(float).copy()
Dose_B = df['drug2.conc'].values.astype(float).copy()
Dose_AB = np.concatenate((Dose_A.reshape(-1,1), Dose_B.reshape(-1,1)),  axis=1)

Effect_B = df[df['drug1.conc'] == 0]['effect'].to_numpy().reshape(-1,1).astype(float)
Effect_A = df[df['drug2.conc'] == 0]['effect'].to_numpy().reshape(-1,1).astype(float)
Dose_A = df[df['drug2.conc']==0]['drug1.conc'].to_numpy().reshape(-1,1).astype(float)
Dose_B = df[df['drug1.conc']==0]['drug2.conc'].to_numpy().reshape(-1,1).astype(float)


# hyperparameters of the priors
A_max  = np.max(Dose_A)
B_max = np.max(Dose_B)

alphaA, betaA = compute_prior_hyperparameters(A_max/100, 0.001*A_max)
alphaB, betaB = compute_prior_hyperparameters(B_max/10, 0.1*B_max)

eff_max_a = np.max(Effect_A)
eff_max_b = np.max(Effect_B)
eff_max = np.max([eff_max_a, eff_max_b])

alpha_var, beta_var = compute_prior_hyperparameters(eff_max, 0.1*eff_max)

#[l1_init, l2_init] = np.meshgrid(np.linspace(np.max(Dose_AB[:,0])/10.0, np.max(Dose_AB[:,0]), 10), np.linspace(np.max(Dose_AB[:,1])/10.0,  np.max(Dose_AB[:,1]), 10))
[l1_init, l2_init] = np.meshgrid(np.linspace(0.0, np.max(Dose_A), 10), np.linspace(0.0,  np.max(Dose_B), 10))
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
        m.kernel.variance_da.assign(alpha_var)

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
plt.title('GP dose-response curve', fontsize=20)
plt.xlabel("$Dose$", fontsize=20), plt.ylabel("Effect", fontsize=20)
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
plt.title('GP dose-response curve', fontsize=20)
plt.xlabel("$Dose$", fontsize=20), plt.ylabel("Effect", fontsize=20)
plt.savefig(drug_name+'_DrugB_2drugs'+'.png')


dim2_A = mean2.iloc[0].to_numpy()
dim2_B = mean2.loc[:][0].to_numpy()

N = 5000

Effect = df['effect'].values.reshape(-1,1).copy()
Dose_A = df['drug1.conc'].values.astype(float).copy()
Dose_B = df['drug2.conc'].values.astype(float).copy()
Dose_AB = np.concatenate((Dose_A.reshape(-1,1), Dose_B.reshape(-1,1)),  axis=1)

Effect_B = df[df['drug1.conc'] == 0]['effect'].to_numpy().reshape(-1,1).astype(float)
Effect_A = df[df['drug2.conc'] == 0]['effect'].to_numpy().reshape(-1,1).astype(float)
Dose_A = df[df['drug2.conc']==0]['drug1.conc'].to_numpy().reshape(-1,1).astype(float)
Dose_B = df[df['drug1.conc']==0]['drug2.conc'].to_numpy().reshape(-1,1).astype(float)

print(df)
print(Dose_A.shape)
print(Dose_B.shape)


data = pd.concat([pd.DataFrame(Dose_AB), pd.DataFrame(Effect)], axis=1)
data.columns = ['Dose_A', 'Dose_B', 'Effect']

X1 = data[data['Dose_B']==0]['Dose_A'].to_numpy().reshape(-1,1).astype(float)
Y1 = data[data['Dose_B']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

X2 = data[data['Dose_A']==0]['Dose_B'].to_numpy().reshape(-1,1).astype(float)
Y2 = data[data['Dose_A']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)


print('X1', X1)
print('X2', X2)


Y_expected_Hand = fit_Hand(N, X1, X2, dim2_A, dim2_B, Dose_A, Dose_B)
Y_expected_Hand_check = fit_Hand(N, X2, X1, dim2_B, dim2_A, Dose_B, Dose_A)

mean_full, Cov_full = predict_in_observations(X1, X2, m)
xv, yv = np.meshgrid(X1, X2)
#result = np.concatenate((xv.reshape(-1,1), yv.reshape(-1,1), (Y_expected_Hand - mean_full).reshape(-1,1)), axis = 1)

print(Y_expected_Hand - Y_expected_Hand_check)

xx_a = np.linspace(np.min(Dose_A), np.max(Dose_A), dim2_A.shape[0]).reshape(-1,1)
xx_b = np.linspace(np.min(Dose_B), np.max(Dose_B), dim2_B.shape[0]).reshape(-1,1)

#v = np.linspace(0.0, 1.0, 10, endpoint=True)

print('Dose_A', Dose_A, Dose_A.shape)
print('Dose_B', Dose_B, Dose_B.shape)
print('Y_expected_Hand', Y_expected_Hand.shape)

plt.figure(figsize=(12, 6))
plt.contourf(np.log(Dose_A.flatten()+np.array(m.kernel.lengthscale_da.value())), np.log(Dose_B.flatten()+np.array(m.kernel.lengthscale_db.value())),Y_expected_Hand)
plt.colorbar()
plt.xlabel('log(dose_a +l_a)', fontsize=20)
plt.ylabel('log(dose_b +l_b)', fontsize=20)
plt.title('Hand model contour', fontsize=20)
plt.savefig(str(drug_name)+'_Hand_contour_result'+'.png')

print(Y_expected_Hand.shape)
print(mean_full.shape)

plt.figure(figsize=(12, 6))
plt.contourf(np.log(Dose_A.flatten()+np.array(m.kernel.lengthscale_da.value())), np.log(Dose_B.flatten()+np.array(m.kernel.lengthscale_db.value())), mean_full)
plt.colorbar()
plt.xlabel('dose_a ', fontsize=20)
plt.ylabel('dose_b', fontsize=20)
plt.title('GP mean contour', fontsize=20)
plt.savefig(str(drug_name)+'_GP_mean'+'.png')


#v = np.linspace(-1.0, 1.0, 10, endpoint=True)
plt.figure(figsize=(12, 6))
v = np.linspace(-0.5, 0.5, 10, endpoint=True)
plt.contourf(np.log(Dose_A.flatten()+np.array(m.kernel.lengthscale_da.value())),
np.log(Dose_B.flatten()+np.array(m.kernel.lengthscale_db.value())), Y_expected_Hand - mean_full, cmap='RdYlGn')
plt.colorbar()
plt.xlabel('log(dose_a +l_a)', fontsize=20)
plt.ylabel('log(dose_b +l_b)', fontsize=20)
plt.title('Difference between GP and Hand-GP', fontsize=20)
plt.savefig(str(drug_name)+'contour_result'+'.png')

# HMC

# num_burnin_steps = ci_niter(1000)
# num_samples = ci_niter(10000)
#
# # Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
# hmc_helper = gpflow.optimizers.SamplingHelper(
#     m.log_posterior_density, m.trainable_parameters
# )
#
# hmc = tfp.mcmc.HamiltonianMonteCarlo(
#     target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=5, step_size=0.3
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
