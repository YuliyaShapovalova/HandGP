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
from gpflow import set_trainable
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


from utilities import (compute_prior_hyperparameters, trapezoidal_area, fit_1d_model, predict_in_observations, fit_Hand,y_exp_Hand1,  y_exp_Hand, y_exp, fit_3d_model, find_nearest, K_log, K_multiplicative)



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_above(my_array, target):
    diff = my_array - target
    mask = np.ma.less_equal(diff, 0)
    # We need to mask the negative differences and zero
    # since we are looking for values above
    if np.all(mask):
        return np.nan # returns None if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

N=50 # number splits in Hand model
#drug_pair = 'LA_bidirectional synergy'
drug_pair = 'LA - bidirectional antagonism'
drug_name = 'LA_bidirectional_synergy'
drug_name = 'LA_bidirectional_antagonism'
#priors = pd.read_csv('priors.txt', sep=' ')

mean_full = pd.read_csv('LA_bidirectional_antagonism_mean_full.csv', sep=',', header=None).to_numpy()

df = pd.read_csv(str(drug_pair)+'.csv')

Effect = df['E'].values.reshape(-1,1).copy()
Dose_A = np.asarray(df['DRUG_A'].values.astype(float).copy()).reshape(-1,1)
Dose_B = np.asarray(df['DRUG_B'].values.astype(float).copy()).reshape(-1,1)
Dose_AB = np.concatenate((Dose_A.reshape(-1,1), Dose_B.reshape(-1,1)), axis=1)

Effect_B = df[df['DRUG_A'] == 0]['E'].to_numpy().reshape(-1,1).astype(float)
Effect_A = df[df['DRUG_B'] == 0]['E'].to_numpy().reshape(-1,1).astype(float)
Dose_A = df[df['DRUG_B']==0]['DRUG_A'].to_numpy().reshape(-1,1).astype(float)
Dose_B = df[df['DRUG_A']==0]['DRUG_B'].to_numpy().reshape(-1,1).astype(float)

# hyperparameters of the priors
A_max  = np.max(Dose_A)
B_max = np.max(Dose_B)

alphaA, betaA = compute_prior_hyperparameters(A_max/5, 0.001*A_max)
alphaB, betaB = compute_prior_hyperparameters(B_max/5, 0.001*B_max)


eff_max_a = np.max(Effect_A)
eff_max_b = np.max(Effect_B)
eff_max = np.max([eff_max_a, eff_max_b])

alpha_var, beta_var = compute_prior_hyperparameters(eff_max, 0.01*eff_max)
alpha_var, beta_var = compute_prior_hyperparameters(75.0/10, 0.01)


# Add extra monotherapeutic data
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


# Optimizae the model

[l1_init, l2_init] = np.meshgrid(np.linspace(.1, 205.0, 10), np.linspace(.1, 205.0, 10))
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
        #set_trainable(m.likelihood.variance, False)

        m.kernel.lengthscale_da.assign(init_lengthscale_da)
        m.kernel.lengthscale_db.assign(init_lengthscale_db)
        m.kernel.variance_da.assign(alpha_var)

        m.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(2.0), np.float64(1.0))
        m.kernel.variance_da.prior = tfp.distributions.Gamma(np.float64(alpha_var), np.float64(beta_var))
        #m.kernel.variance_da.prior = tfp.distributions.HalfNormal(scale=np.float64(2.0))


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
#set_trainable(m.likelihood.variance, False)


m.kernel.lengthscale_da.assign(init_lengthscale_da)
m.kernel.lengthscale_db.assign(init_lengthscale_db)
m.kernel.variance_da.assign(init_var)

# priors
m.likelihood.variance.prior = tfp.distributions.Gamma(np.float64(2.0), np.float64(1.0))
m.kernel.variance_da.prior = tfp.distributions.Gamma(np.float64(alpha_var), np.float64(beta_var))
#m.kernel.variance_da.prior = tfp.distributions.HalfNormal(scale=np.float64(2.0))

m.kernel.lengthscale_da.prior = tfp.distributions.Gamma(np.float64(alphaA), np.float64(betaA))
m.kernel.lengthscale_db.prior = tfp.distributions.Gamma(np.float64(alphaB), np.float64(betaB))


opt = gpflow.optimizers.Scipy()

opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

#m.kernel.lengthscale_da.assign(0.1)

print_summary(m)

# Predict and plot monotherameutic curves


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


mean_full_est = pd.DataFrame(np.asarray(mean2).reshape(Xi.shape))
Cov_full_est = pd.DataFrame(np.asarray(Cov2).reshape(Xi.shape))

dim2_A = mean2.iloc[0].to_numpy()
dim2_B = mean2.loc[:][0].to_numpy()

N = 5000

mean_full_est = pd.DataFrame(mean2_full.reshape(Xi.shape))
Cov_full_est = pd.DataFrame(Cov2_full.reshape(Xi.shape))
#df1 = pd.read_csv('TestData2LA_bidirectional_synergy.csv', sep= ",")
#df2 = pd.read_csv('TestDataLA_bidirectional_synergy.csv', sep = ",")
############################################################################
dim2_A = mean_full_est.iloc[0].to_numpy()
dim2_B = mean_full_est.loc[:][0].to_numpy()

data = pd.concat([pd.DataFrame(Dose_AB), pd.DataFrame(Effect)], axis=1)
data.columns = ['Dose_A', 'Dose_B', 'Effect']

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

############################################################################
mean1 =  dim2_A.copy().reshape(-1,1)
mean2 =  dim2_B.copy().reshape(-1,1)
xx1 = np.linspace(np.min(Dose_A), np.max(Dose_A), mean1.shape[0]).reshape(-1,1)
xx2 = np.linspace(np.min(Dose_B), np.max(Dose_B), mean2.shape[0]).reshape(-1,1)

#d1 = {'fa': mean1, 'fb': mean2, 'xx_a': xx1, 'xx_b': xx2}
d1 = np.concatenate((mean1, mean2, xx1, xx2), axis=1)
print(d1)
df1 = pd.DataFrame(data=d1)
df1.columns = ['fa', 'fb', 'xx_a', 'xx_b']
df1.to_csv('TestData'+str(drug_pair)+'_new'+'.csv')


d2 = np.concatenate((X1, X2, Dose_A, Dose_B), axis=1)
#d2 = {'X1': [X1], 'X2': [X2], 'Dose_A': [Dose_A], 'Dose_B': [Dose_B]}
print(d2)
df2 = pd.DataFrame(data=d2)
df2.columns = ['X1', 'X2', 'Dose_A', 'Dose_B']
df2.to_csv('TestData2'+str(drug_pair)+'_new'+'.csv')
############################################################################


Y_expected_Hand = fit_Hand(N, X1, X2, dim2_A, dim2_B, Dose_A, Dose_B)
Y_expected_Hand_check = fit_Hand(N, X2, X1, dim2_B, dim2_A, Dose_B, Dose_A)

mean_full, Cov_full = predict_in_observations(X1, X2, m)

print(mean_full.shape)


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
v = np.linspace(-100.0, 100.0, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
#cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(),Y_expected_Hand - mean_full,cmap='RdYlGn')
cf = ax.contourf(np.log(Dose_A.flatten()/np.array(m.kernel.lengthscale_da.value())+1), np.log(Dose_B.flatten()/np.array(m.kernel.lengthscale_db.value())+1),Y_expected_Hand - mean_full,cmap='RdYlGn')
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
cf = ax.contourf(np.log(Dose_A.flatten()/np.array(m.kernel.lengthscale_da.value())+1), np.log(Dose_B.flatten()/np.array(m.kernel.lengthscale_db.value())+1),mean_full - Effect.reshape(11,11), v, cmap='YlGnBu')

cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)

plt.xlabel('log($x_1$/$l_1$+1)', fontsize=15)
plt.ylabel('log($x_2$/$l_2$+1)', fontsize=15)
#plt.title("GP residuals", fontsize=20)
plt.savefig(str(drug_name)+'_GP_residuals'+'.png')

#df1 = pd.read_csv('TestData2LA_bidirectional_synergy.csv', sep= ",")
#df2 = pd.read_csv('TestDataLA_bidirectional_synergy.csv', sep = ",")

df1 = pd.read_csv('TestData2LA_bidirectional_synergy_new.csv', sep= ",")
df2 = pd.read_csv('TestDataLA_bidirectional_synergy_new.csv', sep= ",")

print(df1)

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

#indx_dose_final = find_nearest_above(xx1, 0.1)
#print('indx_dose_final ', indx_dose_final)
#print('xx1[indx_dose_final]', xx1[indx_dose_final])

#indx_dose_final = find_nearest_above(mean1, 0.1)
#print('indx_dose_final', indx_dose_final)
#print('mean1[indx_dose_final]', mean1[indx_dose_final])



#Y_expected_Hand = fit_Hand(N, X1, X2, dim2_A, dim2_B, Dose_A, Dose_B)

#print(y_exp_Hand1(N, a, b, mean1, xx1, mean2, xx2))


Y_expected_Hand = fit_Hand(N, X1, X2, mean1, mean2, Dose_A, Dose_B)
#Y_expected_Hand = y_exp_Hand1(N, a, b, mean1, xx1, mean2, xx2)

print('Y_expected_Hand',y_exp_Hand1(N, a, b, mean1, xx1, mean2, xx2))

# v = np.linspace(-1.0, 1.1, 10, endpoint=True)
# plt.figure(figsize=(12, 6))
# plt.contourf(Dose_A.flatten(), Dose_B.flatten(), mean_full - Y_expected_Hand, v, cmap='RdYlGn')
# plt.colorbar()
# plt.savefig(drug_pair+'contour_result'+'.png')


fig, ax = plt.subplots(figsize=(6,6))
v = np.linspace(-1.0, 1.1, 10, endpoint=True)
ax.set_aspect('equal')
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
#cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(), mean_full - Y_expected_Hand, v, cmap='RdYlGn')
cf = ax.contourf(np.log(Dose_A.flatten()/np.array(m.kernel.lengthscale_da.value()+1)), np.log(Dose_B.flatten()/np.array(m.kernel.lengthscale_db.value()+1)),mean_full - Y_expected_Hand, v, cmap='RdYlGn')
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.xlabel('log($x_1$+$l_1$)', fontsize=15)
plt.ylabel('log($x_2$+$l_2$)', fontsize=15)
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(15)

#plt.title("Hand-GP estimated effect", fontsize=20)
plt.savefig(drug_pair+'contour_result'+'.png')

# v = np.linspace(-1.0, 1.1, 10, endpoint=True)
# plt.figure(figsize=(12, 6))
# plt.contourf(Dose_A.flatten(), Dose_B.flatten(), Effect.reshape(X1.shape[0],X1.shape[0]) - Y_expected_Hand, v, cmap='RdYlGn')
# plt.colorbar()
# plt.savefig(drug_pair+'contour_result_obs'+'.png')

# plt.figure(figsize=(12, 6))
# plt.contourf(Dose_A.flatten(), Dose_B.flatten(), mean_full)
# plt.colorbar()
# plt.savefig(drug_pair+'_mean_full_contour_result'+'.png')

fig, ax = plt.subplots(figsize=(6,6))
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
ax.set_aspect('equal')
#cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(), mean_full)
cf = ax.contourf(np.log(Dose_A.flatten()/np.array(m.kernel.lengthscale_da.value()+1)), np.log(Dose_B.flatten()/np.array(m.kernel.lengthscale_db.value()+1)), mean_full)

cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
#plt.xlabel('log($x_1$+$l_1$)', fontsize=15)
#plt.ylabel('log($x_2$+$l_2$)', fontsize=15)
plt.xlabel('log($x_1$/$l_1$+1)', fontsize=15)
plt.ylabel('log($x_2$/$l_2$+1)', fontsize=15)
#plt.title("GP estimated surface", fontsize=20)
plt.savefig(drug_pair+'_mean_full_contour_result'+'.png')


# plt.figure(figsize=(12, 6))
# plt.contourf(Dose_A.flatten(), Dose_B.flatten(),Y_expected_Hand)
# plt.colorbar()
# plt.xlabel('a')
# plt.ylabel('b')
# plt.savefig(drug_pair+'_Hand_contour_result'+'.png')

fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
v = np.linspace(-0.15, 1.05, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
#cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(),Y_expected_Hand,v)
cf = ax.contourf(np.log(Dose_A.flatten()/9.800631439763732+1), np.log(Dose_B.flatten()/9.800779282767849+1),Y_expected_Hand,v)
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
#plt.xlabel('log($x_1$+$l_1$)', fontsize=15)
#plt.ylabel('log($x_2$+$l_2$)', fontsize=15)
plt.xlabel('log($x_1$/$l_1$+1)', fontsize=15)
plt.ylabel('log($x_2$/$l_2$+1)', fontsize=15)
#plt.title("Hand-GP estimated surface", fontsize=18)
plt.savefig(drug_pair+'_Hand_contour_result'+'.png')

fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
v = np.linspace(-0.1, 0.1, 10, endpoint=True)
fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.8)
cf = ax.contourf(np.log(Dose_A.flatten()/9.800631439763732+1), np.log(Dose_B.flatten()/9.800779282767849+1),mean_full - Effect.reshape(11,11), v, cmap='YlGnBu')
#cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(),mean_full - Effect.reshape(11,11), v, cmap='YlGnBu')
cbar = fig.colorbar(cf, ax=ax)

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(15)
#plt.xlabel('log($x_1$+$l_1$)', fontsize=15)
#plt.ylabel('log($x_2$+$l_2$)', fontsize=15)
plt.xlabel('log($x_1$/$l_1$+1)', fontsize=15)
plt.ylabel('log($x_2$/$l_2$+1)', fontsize=15)
#plt.title("GP residuals", fontsize=20)
plt.savefig(drug_pair+'_GP_residuals'+'.png')
exit()
# fig, ax = plt.subplots(figsize=(6,6))
# ax.set_aspect('equal')
# v = np.linspace(-0.1, 0.1, 10, endpoint=True)
# cf = ax.contourf(Dose_A.flatten(), Dose_B.flatten(),Y_expected_Hand - Effect.reshape(11,11), v, cmap='YlGnBu')
# cbar = fig.colorbar(cf, ax=ax)
#
# plt.tick_params(axis='x', labelsize=15)
# plt.tick_params(axis='y', labelsize=15)
#
# for t in cbar.ax.get_yticklabels():
#      t.set_fontsize(15)
#
# plt.title("Hand-GP residuals", fontsize=20)
# plt.savefig(drug_pair+'_HandGP_residuals'+'.png')

#print(priors)
#print(priors[priors['drug_pair']=='AMB.AMB']['a1'].to_numpy()[0])


#HMC
def plot_samples(step_size,samples, parameters, y_axis_label):
    plt.figure(figsize=(8, 4))
    for val, param in zip(samples, parameters):
        plt.plot(tf.squeeze(val), label=param_to_name[param])
    plt.legend(bbox_to_anchor=(1.0, 3.0))
    plt.xlabel("HMC iteration")
    plt.ylabel(y_axis_label)
    plt.savefig('LA_synergy_samples_'+str(step_size)+'.png')

def marginal_samples(step_size, leap_frog, samples, parameters, y_axis_label):
    fig, axes = plt.subplots(1, len(param_to_name), figsize=(15, 3), constrained_layout=True)
    for ax, val, param in zip(axes, samples, parameters):
        ax.hist(np.stack(val).flatten(), bins=20)
        ax.set_title(param_to_name[param])
    fig.suptitle(y_axis_label)
    plt.savefig('LA_synergy_samples_'+str(step_size)+str(leapfrog)+'.png')
    plt.show()

leapfrog_num_and_step_size = np.stack(np.meshgrid([20.0, 15.0, 10.0, 5.0, 1.0], [0.1, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0])).T.reshape(-1, 2)
num_hmc_parameters = leapfrog_num_and_step_size.shape[0]

for j in range(num_hmc_parameters):
    #try:

    num_burnin_steps = ci_niter(1000)
    num_samples = ci_niter(10000)

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
    marginal_samples(step_size,parameter_samples, m.trainable_parameters, "constrained parameter samples")

    #plot_samples(samples, m_full.trainable_parameters, "unconstrained values")
    plot_samples(step_size,parameter_samples, m.trainable_parameters, "constrained parameter values")

    #except:
    #    print('HMC did not work')
