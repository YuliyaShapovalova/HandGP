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
from scipy.stats.distributions import chi2
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from gpflow.utilities import print_summary, positive
np.set_printoptions(suppress=True)
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def compute_prior_hyperparameters(mean, variance):
    # mean equal to variance of empirical standard deviation
    beta = mean/variance
    alpha = mean**2/variance
    return alpha, beta

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

def fit_1d_model(Dose_AB, Effect):
    [l1_init, l2_init] = np.meshgrid(np.linspace(1.0, (np.max(Dose_AB[:,0])-np.min(Dose_AB[:,0]))/10., 10), np.linspace(1.0,  (np.max(Dose_AB[:,1])-np.min(Dose_AB[:,1]))/10., 10))
    l1_init = l1_init.reshape(-1,1)
    l2_init = l2_init.reshape(-1,1)
    Lik_null = np.zeros((100,1))
    Lik_full = np.zeros((100,1))

    Lik_A = np.zeros((100,1))
    Lik_B = np.zeros((100,1))

    data = pd.concat([pd.DataFrame(Dose_AB), pd.DataFrame(Effect)], axis=1)
    data.columns = ['Dose_A', 'Dose_B', 'Effect']

    Dose_A = data[data['Dose_B']==0]['Dose_A'].to_numpy().reshape(-1,1).astype(float)
    Effect_A = data[data['Dose_B']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

    Dose_B = data[data['Dose_A']==0]['Dose_B'].to_numpy().reshape(-1,1).astype(float)
    Effect_B = data[data['Dose_A']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

    for i in range(1,100):
        try:
            init_lengthscale_da = l1_init[i,0]
            init_lengthscale_db = l2_init[i,0]
            init_variance = 1.0
            init_likelihood_variance = 0.01

            k_A = gpflow.kernels.RBF()
            m_A = gpflow.models.GPR(data=(Dose_A, Effect_A), kernel=k_A,  mean_function=None)
            m_A.likelihood.variance.assign(0.01)
            k_A.lengthscales.assign(init_lengthscale_da)
            # priors
            k_A.lengthscale_da.prior = tfp.distributions.InverseGamma(np.float64(8.91924), np.float64(34.5805))
            #m_full.kernel.lengthscale_db.prior = tfp.distributions.InverseGamma(np.float64(8.91924), np.float64(34.5805))
            #m_full.kernel.variance_da.prior = tfp.distributions.InverseGamma(np.float64(1.0), np.float64(1.0))
            opt = gpflow.optimizers.Scipy()
            opt_logs = opt.minimize(m_A.training_loss,
                                    m_A.trainable_variables,method='BFGS',
                                    options=dict(maxiter=100))
            #print_summary(m_full)
            Lik_A[i,0] = np.asarray(m_A.training_loss())

            k_B = gpflow.kernels.RBF()
            m_B = gpflow.models.GPR(data=(Dose_B, Effect_B), kernel=k_B,  mean_function=None)
            m_B.likelihood.variance.assign(0.01)
            k_B.lengthscales.assign(init_lengthscale_db)
            # priors
            k_B.lengthscale_da.prior = tfp.distributions.InverseGamma(np.float64(8.91924), np.float64(34.5805))
            #m_full.kernel.lengthscale_db.prior = tfp.distributions.InverseGamma(np.float64(8.91924), np.float64(34.5805))
            #m_full.kernel.variance_da.prior = tfp.distributions.InverseGamma(np.float64(1.0), np.float64(1.0))
            opt = gpflow.optimizers.Scipy()
            opt_logs = opt.minimize(m_B.training_loss,
                                    m_B.trainable_variables,method='BFGS',
                                    options=dict(maxiter=100))
            #print_summary(m_full)
            Lik_A[i,0] = np.asarray(m_A.training_loss())
            Lik_B[i,0] = np.asarray(m_B.training_loss())


        except:
            Lik_A[i,0] = 'NaN'
            Lik_B[i,0] = 'NaN'
            #print('Cholesky was not successful')

    index_A = np.where(Lik_A == np.nanmin(Lik_A))[0][0]
    index_B = np.where(Lik_B == np.nanmin(Lik_B))[0][0]

    init_lengthscale_da = l1_init[index_A,0]
    init_lengthscale_db = l2_init[index_B,0]

    init_likelihood_variance = 0.01

    k_A = gpflow.kernels.RBF()
    m_A = gpflow.models.GPR(data=(Dose_A, Effect_A), kernel=k_A,  mean_function=None)
    m_A.likelihood.variance.assign(0.01)
    k_A.lengthscales.assign(init_lengthscale_da)
    # priors
    k_A.lengthscales.prior = tfp.distributions.InverseGamma(np.float64(8.91924), np.float64(34.5805))
    #m_full.kernel.lengthscale_db.prior = tfp.distributions.InverseGamma(np.float64(8.91924), np.float64(34.5805))
    #m_full.kernel.variance_da.prior = tfp.distributions.InverseGamma(np.float64(1.0), np.float64(1.0))
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m_A.training_loss,
                            m_A.trainable_variables,method='BFGS',
                            options=dict(maxiter=100))
    #print_summary(m_full)

    k_B = gpflow.kernels.RBF()
    m_B = gpflow.models.GPR(data=(Dose_B, Effect_B), kernel=k_B,  mean_function=None)
    m_B.likelihood.variance.assign(0.01)
    k_B.lengthscales.assign(init_lengthscale_db)
    # priors
    k_B.lengthscales.prior = tfp.distributions.InverseGamma(np.float64(8.91924), np.float64(34.5805))
    #m_full.kernel.lengthscale_db.prior = tfp.distributions.InverseGamma(np.float64(8.91924), np.float64(34.5805))
    #m_full.kernel.variance_da.prior = tfp.distributions.InverseGamma(np.float64(1.0), np.float64(1.0))
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m_B.training_loss,
                            m_B.trainable_variables,method='BFGS',
                            options=dict(maxiter=100))
    #print_summary(m_full)
    Lik_A[i,0] = np.asarray(m_A.training_loss())
    Lik_B[i,0] = np.asarray(m_B.training_loss())

    num_predict = 3000
    #[Xi, Xj] = np.meshgrid(X1, X2)
    xx_A = np.linspace(np.min(Dose_A), np.max(Dose_A),  num_predict).reshape(num_predict, 1)
    xx_B = np.linspace(np.min(Dose_B), np.max(Dose_B),  num_predict).reshape( num_predict, 1)

    # Predict the mean and covariance of the GP fit at the test locations
    mean2_A, Cov2_A = m_A.predict_f(xx_A)
    # Predict the mean and covariance of the GP fit at the test locations
    mean2_B, Cov2_B = m_B.predict_f(xx_B)

    mean2_A = np.asarray(mean2_A)
    Cov2_A = np.asarray(Cov2_A)

    mean2_B = np.asarray(mean2_B)
    Cov2_B = np.asarray(Cov2_B)



    return  m_A, mean2_A, m_B, mean2_B


def predict_in_observations(X1, X2, m_full):
    #[Xi, Xj] = np.meshgrid(X1, X2)
    [Xi, Xj] = np.meshgrid(X1,X2)

    # We need to augument our test space to be a list of coordinates for input to the GP
    Xnew2 = np.vstack((Xi.ravel(), Xj.ravel())).T # Change our input grid to list of coordinates

    # Predict the mean and covariance of the GP fit at the test locations
    mean2_full, Cov2_full = m_full.predict_f(Xnew2)

    mean2_full = np.asarray(mean2_full)
    Cov2_full = np.asarray(Cov2_full)

    return  mean2_full.reshape(Xi.shape), Cov2_full.reshape(Xi.shape)


def fit_Hand(N, X1, X2, dim2_A, dim2_B, Dose_A, Dose_B):
    Y_expected_Hand = np.zeros((X1.shape[0],X1.shape[0]))
    #Y_expected_ab = np.zeros((8,8))
    #Y_expected_ba = np.zeros((8,8))

    xv, yv = np.meshgrid(X1, X2)

    mean1 =  dim2_A.copy().reshape(-1,1)
    mean2 =  dim2_B.copy().reshape(-1,1)
    xx1 = np.linspace(np.min(Dose_A), np.max(Dose_A), mean1.shape[0]).reshape(-1,1)
    xx2 = np.linspace(np.min(Dose_B), np.max(Dose_B), mean2.shape[0]).reshape(-1,1)

    # new based on 2D
    for i in range(0,X1.shape[0]):
        for j in range(0,X1.shape[0]):
            Y_expected_Hand[i,j] = y_exp_Hand1(N, xv[i,j], yv[i,j], mean1, xx1, mean2, xx2)
            #print(xv[i,j])
            #print(yv[i,j])

    return Y_expected_Hand

def y_exp_Hand1(N, a, b, f_a, xx_a, f_b, xx_b):
    # Outputs expected effect for Hand model
    # a and b are vetors of partitions of the doses a and b

    ### For a dose b1 find the effect f(b)

    dose_splits_drug_a = int(find_nearest_above(xx_a, a))
    dose_splits_drug_b = int(find_nearest_above(xx_b, b))

    #if(dose_splits_drug_a>4):
    #    dose_splits_drug_a = int(find_nearest_above(xx_a, a)/4)
    #    print('Wear shorts.')

    #if(dose_splits_drug_b > 4):
    #    dose_splits_drug_b = int(find_nearest_above(xx_b, b)/4)
    #    print('Wear shorts.')


    #print('dose_a', a, 'dose__b',b)
    #print('dose_splits_drug_a', dose_splits_drug_a, 'dose_splits_drug_b', dose_splits_drug_b)

    #dose_splits_drug_a = 40
    #dose_splits_drug_b = 40

    #dose_splits_drug_a = 2000
    #dose_splits_drug_b = 2000

    #N = dose_splits_drug_a.copy()

    a_partition = np.repeat(a/dose_splits_drug_a, [dose_splits_drug_a], axis=0)
    b_partition = np.repeat(b/dose_splits_drug_b, [dose_splits_drug_b], axis=0)
    #print('a_partition', a_partition)
    #print('b_partition', b_partition)
    #print(np.sum(a_partition, axis=0))
    #print(np.sum(b_partition, axis=0))

    #print('a_partition',a_partition)

    total_dose_a = np.repeat(0.0, [dose_splits_drug_a], axis=0)
    total_dose_b = np.repeat(0.0, [dose_splits_drug_b], axis=0)

    indx_dose_a = np.repeat(0.0, [dose_splits_drug_a], axis=0)
    indx_dose_b = np.repeat(0.0, [dose_splits_drug_b], axis=0)

    count_dose_b = 1.0
    count_dose_a = 1.0

    ### For index 0
    ## Apply dose b[0]
    i = 0
    j = 0
    total_dose_b[j] = b_partition[j]
    # Find index of the dose
    indx_dose_b[j] = find_nearest_above(xx_b, b_partition[j])

    # Find corresponding effect
    Effect_reached_b = np.asarray(f_b[int(indx_dose_b[j])])[0] # find the effect b0 reaches
    ### Find dose An for which A reaches effect Eb
    near_effect = find_nearest_above(f_a, Effect_reached_b)   # find the effect for GP predicted for drug A which is close to effect
                                                        # reached by b1
    #indx_effect = np.where(np.isclose(f_a, near_effect))[0][0] # fine the index of this effect
    indx_effect = find_nearest_above(f_a, Effect_reached_b)

    total_dose_a[i] = xx_a[indx_effect] + a_partition[i]       # add the dose a1 to the effect already reached by b1

    # Find closest effect for total dose
    indx_dose_final = find_nearest_above(xx_a, total_dose_a[i])
    # Finally the effect for the total dose
    total_effect = np.asarray(f_a[indx_dose_final])[0]             # find total effect reached
    i = i + 1
    j = j + 1
    #print('Total effect at 0', total_effect)

    while (i < dose_splits_drug_a and j < dose_splits_drug_b):
    #while (total_dose_b[j-1] < b and total_dose_a[i-1] < a):
        #print('i', i)
        #print('j', j)
        #print('total_dose_b[j]', total_dose_b[j])
        #print('total_dose_a[j]', total_dose_a[i])

        near_effect = find_nearest(f_b, total_effect)
        indx_dose = np.where(np.isclose(f_b, near_effect))[0][0] # fine the index of this effect

        #indx_dose = find_nearest_above(f_b, total_effect) # Here it was breaking, why can't we use above?
        #print('indx_dose', indx_dose)
        #print('total_effect', total_effect)

        total_dose_b[j] = xx_b[int(indx_dose)] + count_dose_b*b_partition[j]

        # Find index of the dose
        indx_dose_b[j] = find_nearest_above(xx_b, total_dose_b[j])
        #print('indx_dose_b[j]', indx_dose_b[j], 'indx_dose_b[j-1]', indx_dose_b[j-1])
        if(indx_dose_b[j-1] == indx_dose_b[j]):
            count_dose_b =  count_dose_b + 1.0
        else:
            count_dose_b = 1.0

        # Find corresponding effect
        Effect_reached_b = np.asarray(f_b[int(indx_dose_b[j])])[0] # find the effect b0 reaches
        #print('Effect_reached_b', Effect_reached_b)
        ### Find dose An for which A reaches effect Eb
        near_effect = find_nearest(f_a, Effect_reached_b)   # find the effect for GP predicted for drug A which is close to effect
                                                            # reached by b1
        indx_effect = np.where(np.isclose(f_a, near_effect))[0][0] # fine the index of this effect
        #indx_effect = find_nearest_above(f_a, Effect_reached_b)

        total_dose_a[i] = xx_a[indx_effect] + count_dose_a*a_partition[i]       # add the dose a1 to the effect already reached by b1

        # Find closest effect for total dose
        indx_dose_a[i] = find_nearest_above(xx_a, total_dose_a[i])

        if(indx_dose_a[i-1] == indx_dose_a[i]):
            count_dose_a =  count_dose_a + 1.0
        else:
            count_dose_a = 1.0

        # Finally the effect for the total dose
        total_effect = np.asarray(f_a[int(indx_dose_a[i])])[0]             # find total effect reached
        #print('f_a[int(indx_dose_a[i])', np.asarray(f_a[int(indx_dose_a[i])])[0])
        #print('total_effect', total_effect)
        i = i + 1
        j = j + 1

    if (dose_splits_drug_a > dose_splits_drug_b):
        #a_remaining_dose = a - total_dose_a[i-1]
        a_remaining_dose = dose_splits_drug_a * a_partition[0] - dose_splits_drug_b * a_partition[0]
        # Apply remaining dose
        total_dose_a_final = total_dose_a[i-1] + a_remaining_dose
        indx_dose_a_final = find_nearest_above(xx_a, total_dose_a_final)
        total_effect = np.asarray(f_a[indx_dose_a_final])[0]

    elif(dose_splits_drug_a < dose_splits_drug_b):
        #b_remaining_dose = b - total_dose_b[j]
        b_remaining_dose = dose_splits_drug_b * b_partition[0] - dose_splits_drug_a * b_partition[0]
        total_dose_b_final = total_dose_b[j-1] + b_remaining_dose
        indx_dose_b_final  = find_nearest_above(xx_b, total_dose_b_final)
        total_effect = np.asarray(f_b[indx_dose_b_final])[0]
    else:
        print("Dose splits are equal")
    #print('total_dose_a', total_dose_a)
    #print('total_dose_b', total_dose_b)
    #print('a', a)
    #print('b', b)
    #print('indx_dose_a', indx_dose_a)
    #print('indx_dose_b', indx_dose_b)
    #print('a', a, ' dose_splits_drug_a * a_partition[0]',  dose_splits_drug_a * a_partition[0])
    #print('b', b, ' dose_splits_drug_b * b_partition[0]',  dose_splits_drug_b * b_partition[0])
    #print('dose_splits_drug_a ', dose_splits_drug_a )
    #print('dose_splits_drug_b ', dose_splits_drug_b )
    return(total_effect)


def y_exp_Hand2(N, a, b, f_a, xx_a, f_b, xx_b):
    # Outputs expected effect for Hand model
    # a and b are vetors of partitions of the doses a and b

    ### For a dose b1 find the effect f(b)

    dose_splits_drug_a = int(find_nearest_above(xx_a, a))
    dose_splits_drug_b = int(find_nearest_above(xx_b, b))

    #if(dose_splits_drug_a>4):
    #    dose_splits_drug_a = int(find_nearest_above(xx_a, a)/4)
    #    print('Wear shorts.')

    #if(dose_splits_drug_b > 4):
    #    dose_splits_drug_b = int(find_nearest_above(xx_b, b)/4)
    #    print('Wear shorts.')


    #print('dose_a', a, 'dose__b',b)
    #print('dose_splits_drug_a', dose_splits_drug_a, 'dose_splits_drug_b', dose_splits_drug_b)

    #dose_splits_drug_a = 40
    #dose_splits_drug_b = 40

    #dose_splits_drug_a = 2000
    #dose_splits_drug_b = 2000

    #N = dose_splits_drug_a.copy()

    a_partition = np.repeat(a/dose_splits_drug_a, [dose_splits_drug_a], axis=0)
    b_partition = np.repeat(b/dose_splits_drug_b, [dose_splits_drug_b], axis=0)
    #print('a_partition', a_partition)
    #print('b_partition', b_partition)
    #print(np.sum(a_partition, axis=0))
    #print(np.sum(b_partition, axis=0))

    #print('a_partition',a_partition)

    total_dose_a = np.repeat(0.0, [dose_splits_drug_a], axis=0)
    total_dose_b = np.repeat(0.0, [dose_splits_drug_b], axis=0)

    indx_dose_a = np.repeat(0.0, [dose_splits_drug_a], axis=0)
    indx_dose_b = np.repeat(0.0, [dose_splits_drug_b], axis=0)

    count_dose_b = 1.0
    count_dose_a = 1.0

    ### For index 0
    ## Apply dose b[0]
    i = 0
    j = 0
    total_dose_b[j] = b_partition[j]
    # Find index of the dose
    indx_dose_b[j] = find_nearest_above(xx_b, b_partition[j])

    # Find corresponding effect
    Effect_reached_b = np.asarray(f_b[int(indx_dose_b[j])])[0] # find the effect b0 reaches
    ### Find dose An for which A reaches effect Eb
    near_effect = find_nearest_above(f_a, Effect_reached_b)   # find the effect for GP predicted for drug A which is close to effect
                                                        # reached by b1
    #indx_effect = np.where(np.isclose(f_a, near_effect))[0][0] # fine the index of this effect
    indx_effect = find_nearest_above(f_a, Effect_reached_b)

    total_dose_a[i] = xx_a[indx_effect] + a_partition[i]       # add the dose a1 to the effect already reached by b1

    # Find closest effect for total dose
    indx_dose_final = find_nearest_above(xx_a, total_dose_a[i])
    # Finally the effect for the total dose
    total_effect = np.asarray(f_a[indx_dose_final])[0]             # find total effect reached
    i = i + 1
    j = j + 1
    #print('Total effect at 0', total_effect)

    while (i < dose_splits_drug_a and j < dose_splits_drug_b):
    #while (total_dose_b[j-1] < b and total_dose_a[i-1] < a):
        #print('i', i)
        #print('j', j)
        #print('total_dose_b[j]', total_dose_b[j])
        #print('total_dose_a[j]', total_dose_a[i])

        near_effect = find_nearest(f_b, total_effect)
        indx_dose = np.where(np.isclose(f_b, near_effect))[0][0] # fine the index of this effect

        #indx_dose = find_nearest_above(f_b, total_effect) # Here it was breaking, why can't we use above?
        #print('indx_dose', indx_dose)
        #print('total_effect', total_effect)

        total_dose_b[j] = xx_b[int(indx_dose)] #+ count_dose_b*b_partition[j]

        # Find index of the dose
        indx_dose_b[j] = find_nearest_above(xx_b, total_dose_b[j])
        #print('indx_dose_b[j]', indx_dose_b[j], 'indx_dose_b[j-1]', indx_dose_b[j-1])
        if(indx_dose_b[j-1] == indx_dose_b[j]):
            count_dose_b =  count_dose_b + 1.0
        else:
            count_dose_b = 1.0

        # Find corresponding effect
        Effect_reached_b = np.asarray(f_b[int(indx_dose_b[j])])[0] # find the effect b0 reaches
        #print('Effect_reached_b', Effect_reached_b)
        ### Find dose An for which A reaches effect Eb
        near_effect = find_nearest(f_a, Effect_reached_b)   # find the effect for GP predicted for drug A which is close to effect
                                                            # reached by b1
        indx_effect = np.where(np.isclose(f_a, near_effect))[0][0] # fine the index of this effect
        #indx_effect = find_nearest_above(f_a, Effect_reached_b)

        total_dose_a[i] = xx_a[indx_effect] #+ count_dose_a*a_partition[i]       # add the dose a1 to the effect already reached by b1

        # Find closest effect for total dose
        indx_dose_a[i] = find_nearest_above(xx_a, total_dose_a[i])

        if(indx_dose_a[i-1] == indx_dose_a[i]):
            count_dose_a =  count_dose_a + 1.0
        else:
            count_dose_a = 1.0

        # Finally the effect for the total dose
        total_effect = np.asarray(f_a[int(indx_dose_a[i])])[0]             # find total effect reached
        #print('f_a[int(indx_dose_a[i])', np.asarray(f_a[int(indx_dose_a[i])])[0])
        #print('total_effect', total_effect)
        i = i + 1
        j = j + 1

    if (dose_splits_drug_a > dose_splits_drug_b):
        #a_remaining_dose = a - total_dose_a[i-1]
        a_remaining_dose = dose_splits_drug_a * a_partition[0] - dose_splits_drug_b * a_partition[0]
        # Apply remaining dose
        total_dose_a_final = total_dose_a[i-1] + a_remaining_dose
        indx_dose_a_final = find_nearest_above(xx_a, total_dose_a_final)
        total_effect = np.asarray(f_a[indx_dose_a_final])[0]

    elif(dose_splits_drug_a < dose_splits_drug_b):
        #b_remaining_dose = b - total_dose_b[j]
        b_remaining_dose = dose_splits_drug_b * b_partition[0] - dose_splits_drug_a * b_partition[0]
        total_dose_b_final = total_dose_b[j-1] + b_remaining_dose
        indx_dose_b_final  = find_nearest_above(xx_b, total_dose_b_final)
        total_effect = np.asarray(f_b[indx_dose_b_final])[0]
    else:
        print("Dose splits are equal")
    #print('total_dose_a', total_dose_a)
    #print('total_dose_b', total_dose_b)
    #print('a', a)
    #print('b', b)
    #print('indx_dose_a', indx_dose_a)
    #print('indx_dose_b', indx_dose_b)
    #print('a', a, ' dose_splits_drug_a * a_partition[0]',  dose_splits_drug_a * a_partition[0])
    #print('b', b, ' dose_splits_drug_b * b_partition[0]',  dose_splits_drug_b * b_partition[0])
    #print('dose_splits_drug_a ', dose_splits_drug_a )
    #print('dose_splits_drug_b ', dose_splits_drug_b )
    return(total_effect)



def y_exp_Hand(N, a, b, f_a, xx_a, f_b, xx_b):
    # Outputs expected effect for Hand model
    # a and b are vetors of partitions of the doses a and b
    #print('N',N)
    #print('Total dose a + b', a+b)
    #N = int(a/(xx_a[0]-xx_a[1]))
    #N=1000
    ### For a dose b1 find the effect f(b)

    indx_N_a = find_nearest_above(xx_a, a)
    indx_N_b = find_nearest_above(xx_b, b)
    print('indx_N_a', indx_N_a, 'indx_N_b', indx_N_b)

    N = indx_N_a



    a = np.repeat(a/N, [N], axis=0)
    b = np.repeat(b/N, [N], axis=0)
    GP_size = f_a.shape[0]
    print('a', a)
    print('b', b)
    total_dose_a = np.repeat(0.0, [N], axis=0)
    total_dose_b = np.repeat(0.0, [N], axis=0)
    #total_dose = np.repeat(0.0, [N], axis=0)
    print(0)
    ### For index 0
    ## Apply dose b[0]

    # Find index of the dose
    #near_b_dose = find_nearest_above(xx_b, b[0]) # find closest dose to b1 in the GP predicted points

    total_dose_b[0] = b[0]
    #total_dose[0] = b[0]

    indx_dose = find_nearest_above(xx_b, b[0])
    print('xx_b[indx_dose]', xx_b[indx_dose], 'b[0]', b[0])
    # Find corresponding effect
    Eb = np.asarray(f_b[indx_dose])[0] # find the effect b0 reaches
    print('Eb', Eb)
    ### Find dose An for which A reaches effect Eb
    near_effect = find_nearest(f_a, Eb)   # find the effect for GP predicted for drug A which is close to effect
    print('near_effect', near_effect, 'Eb', Eb)                                         # reached by b1
    indx_effect = np.where(np.isclose(f_a, near_effect))[0][0] # fine the index of this effect
    print('indx_efffect', indx_effect, 'f_a[indx_effect]', f_a[indx_effect])
    total_dose_a[0] = xx_a[indx_effect] + a[0]       # add the dose a1 to the effect already reached by b1
    print('xx_a[indx_effect]', xx_a[indx_effect], 'a[0]', a[0])
    print('total_dose_a[0]', total_dose_a[0])

    #total_dose[0] = b[0] + a[0]

    # Find closest effect for total dose
    near_dose = find_nearest_above(xx_a, total_dose_a[0]) # find dose a1 + b1 on the scale of a
    indx_dose_final = find_nearest_above(xx_a, total_dose_a[0])
    #if(np.isnan(indx_dose_final)):
    #    indx_dose_final = xx_a.shape[0]-1
    print('xx_a[indx_dose_final]', xx_a[indx_dose_final], 'total_dose_a[0]', total_dose_a[0])
    # Finally the effect for the total dose
    total_effect = np.asarray(f_a[indx_dose_final])[0]             # find total effect reached
    print('total_effect', total_effect)


    for i in range(1, N):
        print(i)

        ### Find dose B for which (a_{i-1}+b_{i-1}) reached total effect
        near_effect = find_nearest(f_b, total_effect)
        #print('near_effect (in fb)', near_effect, 'total effect', total_effect)
        #indx_effect = np.where(np.isclose(f_b, near_effect))[0][0] # fine the index of this effect
        indx_effect = find_nearest_above(f_b, total_effect) # Here it was breaking, why can't we use above?
        #print('total_effect, indx_effect',  total_effect, indx_effect)


        #f_b  = f_b[(indx_effect+1):GP_size] # reduce vector f_b to avoid numerical issues
        #print('indx_effect', indx_effect)
        # Find closest effect for total dose
        #if(np.isnan(indx_effect)):
        #    indx_effect = int(xx_b.shape[0]-1)

        total_dose_b[i] = xx_b[indx_effect] + b[i]
        #total_dose_b[i] = total_dose_b[i-1] + b[i]
        #print('xx_b[indx_effect]', xx_b[indx_effect], 'b[i]', b[i])

        #print('total_dose_b', total_dose_b[i])
        indx_dose_final = find_nearest_above(xx_b, total_dose_b[i])  # find index of this dose

        #xx_b  = xx_b[(indx_effect+1):GP_size] # now reduce the vector xx_b to reduce numerical issues
        # Finally the effect for the total dose
        #if(np.isnan(indx_dose_final)):
        #    indx_dose_final = int(xx_a.shape[0]-1)
        #print('indx_dose_final', indx_dose_final, 'xx_b[indx_dose_final]', xx_b[indx_dose_final])
        total_effect = np.asarray(f_b[indx_dose_final])[0]             # find total effect reached
        #print('total effect', total_effect)
        # Do the same for drug A
        near_effect = find_nearest(f_a, total_effect)
        #print('near_effect', near_effect, 'total_effect', total_effect)
        indx_effect = np.where(np.isclose(f_a, near_effect))[0][0] # fine the index of this effect
        #print('indx_effect', indx_effect, 'near_effect', near_effect)
        indx_effect = find_nearest_above(f_a, near_effect)

        #f_a  = f_a[(indx_effect+1):GP_size] # reduce the vector to avoid numerical issues
        #if(np.isnan(indx_effect)):
        #    indx_effect = int(xx_a.shape[0]-1) # Change to GP_size

        #print('indx_effect', indx_effect)
        #print('f_a[indx_effect]', f_a[indx_effect], 'near_effect', near_effect)

        # Find closest effect for total dose

        #print('indx_effect', indx_effect)
        total_dose_a[i] = xx_a[indx_effect] + a[i]
        #print('xx_a[indx_effect]', xx_a[indx_effect], 'a[i]', a[i])
        #total_dose_a[i] = total_dose_a[i-1] + a[i]
        #print('total_dose_a', total_dose_a[i])

        # Find closest effect for total dose

        near_dose = find_nearest_above(xx_a, total_dose_a[i]) # find dose a1 + b1 on the scale of a
        indx_dose_final = find_nearest_above(xx_a, total_dose_a[i])  # find index of this dose

        #xx_a  = xx_a[(indx_effect+1):GP_size] # reduce xx_a vector to avoid numerical issues
        #if(np.isnan(indx_dose_final)):
        #    indx_dose_final = xx_a.shape[0]-1 # Change to GP_size

        # Finally the effect for the total dose
        total_effect = np.asarray(f_a[indx_dose_final])[0]             # find new total effect reached
        #print('total_effect', total_effect)

    print('total_dose_a', total_dose_a)
    print('total_dose_b', total_dose_b)
    #print('fa', f_a)
    #print('fb', f_b)
    return(total_effect)


def y_exp(a, b, f_a, xx_a, f_b, xx_b):
    # Gives y_exp(a+b) fot Tallarida
    ### For a dose B find the effect Eb = f(b)
    # Find index of the dose
    near_b_dose = find_nearest(xx_b, b)[0]
    indx_dose = np.where(np.isclose(xx_b, near_b_dose ))
    # Find corresponding effect
    Eb = np.asarray(f_b[indx_dose[0][0]])[0]

    ### Find dose An for which A reaches effect Eb
    near_effect = find_nearest(f_a, Eb)[0]
    indx_effect = np.where(np.isclose(f_a, near_effect))[0][0]
    total_dose = xx_a[indx_effect] + a
    # Find closest effect for total dose
    near_dose = find_nearest(xx_a, total_dose)[0]
    indx_dose_final = np.where(np.isclose(xx_a, near_dose))[0][0]
    # Finally the effect for the total dose
    total_effect = np.asarray(f_a[indx_dose_final])[0]

    return(total_effect)


def fit_3d_model(Dose_AB, Effect):
    [l1_init, l2_init] = np.meshgrid(np.linspace(1.0, (np.max(Dose_AB[:,0])-np.min(Dose_AB[:,0]))/10., 10), np.linspace(1.0,  (np.max(Dose_AB[:,1])-np.min(Dose_AB[:,1]))/10., 10))
    l1_init = l1_init.reshape(-1,1)
    l2_init = l2_init.reshape(-1,1)
    Lik_null = np.zeros((100,1))
    Lik_full = np.zeros((100,1))

    for i in range(1,100):
        try:
            init_lengthscale_da = l1_init[i,0]
            init_lengthscale_db = l2_init[i,0]
            init_variance = 1.0
            init_likelihood_variance = 0.01

            k_full = K_multiplicative()
            m_full = gpflow.models.GPR(data=(Dose_AB, Effect), kernel=k_full,  mean_function=None)
            m_full.likelihood.variance.assign(0.01)
            m_full.kernel.lengthscale_da.assign(init_lengthscale_da)
            m_full.kernel.lengthscale_db.assign(init_lengthscale_db)
            # priors
            #m_full.kernel.lengthscale_da.prior = tfp.distributions.InverseGamma(np.float64(8.91924), np.float64(34.5805))
            #m_full.kernel.lengthscale_db.prior = tfp.distributions.InverseGamma(np.float64(8.91924), np.float64(34.5805))

            m_full.kernel.lengthscale_da.prior = tfp.distributions.Normal(loc = np.float64(15.0),scale=np.float64(2.0))
            m_full.kernel.lengthscale_db.prior = tfp.distributions.Normal(loc = np.float64(15.0),scale=np.float64(2.0))


            m_full.kernel.variance_da.prior = tfp.distributions.InverseGamma(np.float64(1.0), np.float64(1.0))
            opt = gpflow.optimizers.Scipy()
            opt_logs = opt.minimize(m_full.training_loss,
                                    m_full.trainable_variables,method='BFGS',
                                    options=dict(maxiter=100))
            #print_summary(m_full)
            Lik_full[i,0] = np.asarray(m_full.log_likelihood())
        except:
            Lik_full[i,0] = 'NaN'
            #print('Cholesky was not successful')

    #index = np.where(Lik_full == np.nanmin(Lik_full))[0][0]
    index = np.where(Lik_full == np.nanmax(Lik_full))[0][0]

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
    #m_null.kernel.lengthscale_da.prior = tfp.distributions.InverseGamma(np.float64(1.0), np.float64(1.0))
    #m_null.kernel.lengthscale_db.prior = tfp.distributions.InverseGamma(np.float64(1.0), np.float64(1.0))
    #m_null.kernel.variance_da.prior = tfp.distributions.InverseGamma(np.float64(1.0), np.float64(1.0))

    #m_full.kernel.lengthscale_da.prior = tfp.distributions.HalfNormal(np.float64(np.max(Dose_AB[:,0])-np.min(Dose_AB[:,0])))
    #m_full.kernel.lengthscale_db.prior = tfp.distributions.HalfNormal(np.float64(np.max(Dose_AB[:,1])-np.min(Dose_AB[:,1])))
    #m_full.kernel.lengthscale_da.prior = tfp.distributions.Uniform(low=dfloat(15.0), high=dfloat(70.0))
    #m_full.kernel.lengthscale_db.prior = tfp.distributions.Uniform(low=dfloat(15.0), high=dfloat(70.0))
    m_full.kernel.lengthscale_da.prior = tfp.distributions.Normal(loc = np.float64(15.0),scale=np.float64(2.0))
    m_full.kernel.lengthscale_db.prior = tfp.distributions.Normal(loc = np.float64(15.0),scale=np.float64(2.0))

    m_full.kernel.variance_da.prior = tfp.distributions.HalfNormal(np.float64(1.0))

    #m_full.kernel.lengthscale_da.prior = tfp.distributions.InverseGamma(np.float64(8.91924), np.float64(34.5805))
    #m_full.kernel.lengthscale_db.prior = tfp.distributions.InverseGamma(np.float64(8.91924), np.float64(34.5805))
    #m_full.kernel.variance_da.prior = tfp.distributions.InverseGamma(np.float64(1.0), np.float64(1.0))

    #m_null.likelihood.variance.trainable = False

    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m_full.training_loss,
                        m_full.trainable_variables, method='BFGS',
                        options=dict(maxiter=1000))
    print_summary(m_full)

    num_predict = 3000
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

    return m_full, mean_full_est, Cov_full_est



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
        #return np.nan # returns None if target is greater than any value
        return int(my_array.shape[0]-1) # returns max value if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

import numpy as np
import scipy.spatial

def main(xyz):
    area_underneath = trapezoidal_area(xyz)
    print(area_underneath)

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

#main()

class K_log(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__(active_dims=[0])
        self.variance = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
        self.lengthscale = gpflow.Parameter(2.0, transform=gpflow.utilities.positive())

    def K(self, X, X2=None, presliced=None):
        if X2 is None:
            X2 = X
        n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        #X = tf.math.log(X+self.lengthscale_da)
        X_logscaled = tf.math.log(X/self.lengthscale+1.0)
        #X_logscaled[:,1] = tf.math.log(X[:,1]/self.lengthscale_db+1)
        X2_logscaled = tf.math.log(X2/self.lengthscale+1.0)
        #X2_logscaled[:,1] = tf.math.log(X2[:,1]/self.lengthscale_db+1)

        X_matrix = tf.tile(tf.reshape(X_logscaled,(n,1)), (1,m))
        X_matrix_tr = tf.tile(tf.reshape(X2_logscaled, (m,1)), (1,n))
        X_matrix_tr = tf.transpose(X_matrix_tr)

        diff_X = X_matrix - X_matrix_tr

        K_dada = tf.exp(-0.5*(diff_X)**2)

        return self.variance*K_dada

    def K_diag(self, X, presliced=None):
        n = tf.cast(tf.shape(X)[0], dtype=tf.int32)
        diag = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance)), (-1,))
        #diag_db = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_db)), (-1,))
        return diag


def log_kernel_new(X, X2, lengthscale, variance):
    n = np.asscalar(np.array(X.shape[0]).astype(int))
    m = np.asscalar(np.array(X2.shape[0]).astype(int))

    X = X/lengthscale+1
    X2 = X2/lengthscale+1

    X_l = X + lengthscale
    X2_l = X2 + lengthscale

    X_matrix = np.tile(X[0:n], (1,m)).reshape(n,m)
    X_matrix_tr = np.tile(X2[0:m], (1,n)).reshape(m,n)
    X_matrix_tr = np.transpose(X_matrix_tr)

    X_l_matrix = np.tile(X_l[0:n], (1,m)).reshape(n,m)
    X_l_matrix_tr = np.tile(X2_l[0:m], (1,n)).reshape(m,n)
    X_l_matrix_tr = np.transpose(X_l_matrix_tr)

    part1 = -2.06637*variance**2
    part2 = 0.5*lengthscale**2*X_l_matrix_tr**2*erf(-0.707107*np.log(X_matrix)+0.707107*np.log(X_matrix_tr)+0.707107)
    part3 = 0.5*lengthscale**2*X_l_matrix**2*erf(-0.707107*np.log(X_matrix)-0.707107*np.log(X_matrix_tr)+0.707107)

    K = part1*(part2+part3)

    return(K)

class K_sqexp_kernel(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
        self.lengthscale = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())

    def K(self, X, X2=None, presliced=None):
        if X2 is None:
            X2 = X
        n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        #X_logscaled = X/self.lengthscale+1
        #X2_logscaled = X2/self.lengthscale+1

        #X_l = X + self.lengthscale
        #X2_l = X2 + self.lengthscale

        #X_logscaled = tf.stack([X_logscaled_dim0, X_logscaled_dim1], axis=1)
        #X2_logscaled = tf.stack([X2_logscaled_dim0, X2_logscaled_dim1], axis=1)

        X_matrix = tf.tile(tf.reshape(X[0:n],(n,1)), (1,m))
        X_matrix_tr = tf.tile(tf.reshape(X2[0:m], (m,1)), (1,n))
        X_matrix_tr = tf.transpose(X_matrix_tr)
        diff = X_matrix - X_matrix_tr

        part1 = -1.25331*self.variance**2*self.lengthscale
        part2 = diff.tf.math.erf(0.7*diff/self.lengthscale)+0.7*self.lengthscale*tf.math.exp((-0.5*th.math.square(X_matrix)+X_matrix*Xmatrix_tr -0.5*tf.square(X_matrix_tr))/self.lengthscale**2)

        return 100+part1*part2

    def K_diag(self, X, presliced=None):
        n = tf.cast(tf.shape(X)[0], dtype=tf.int32)
        m = tf.cast(tf.shape(X)[0], dtype=tf.int32)

        X = X/self.lengthscale+1
        X_l = X + self.lengthscale

        part1 = -2.06637*self.variance**2
        part2 = 0.5*self.lengthscale**2*X_l**2*tf.math.erf(-0.707107*tf.math.log(X)+0.707107*tf.math.log(X)+0.707107)
        part3 = 0.5*self.lengthscale**2*X_l**2*tf.math.erf(-0.707107*tf.math.log(X)-0.707107*tf.math.log(X)+0.707107)

        #diag = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(part1*(part2+part3))), (-1,))
        diag = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance)), (-1,))

        return diag

class K_new_kernel(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
        self.lengthscale = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())

    def K(self, X, X2=None, presliced=None):
        if X2 is None:
            X2 = X
        n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        X_logscaled = X/self.lengthscale+1
        X2_logscaled = X2/self.lengthscale+1

        X_l = X + self.lengthscale
        X2_l = X2 + self.lengthscale

        #X_logscaled = tf.stack([X_logscaled_dim0, X_logscaled_dim1], axis=1)
        #X2_logscaled = tf.stack([X2_logscaled_dim0, X2_logscaled_dim1], axis=1)

        X_matrix = tf.tile(tf.reshape(X_logscaled[0:n],(n,1)), (1,m))
        X_matrix_tr = tf.tile(tf.reshape(X2_logscaled[0:m], (m,1)), (1,n))
        X_matrix_tr = tf.transpose(X_matrix_tr)

        X_l_matrix = tf.tile(tf.reshape(X_l[0:n],(n,1)), (1,m))
        X_l_matrix_tr = tf.tile(tf.reshape(X2_l[0:m], (m,1)), (1,n))
        X_l_matrix_tr = tf.transpose(X_l_matrix_tr)

        part1 = -2.06637*self.variance**2
        part2 = 0.5*self.lengthscale**2*tf.math.square(X_matrix_tr)*tf.math.erf(-0.707107*tf.math.log(X_matrix)+0.707107*tf.math.log(X_matrix_tr)+0.707107)
        part3 = 0.5*tf.math.square(X_l_matrix)*tf.math.erf(0.707107*tf.math.log(X_matrix)-0.707107*tf.math.log(X_matrix_tr)+0.707107)

        return part1*(part2+part3)

    def K_diag(self, X, presliced=None):
        n = tf.cast(tf.shape(X)[0], dtype=tf.int32)
        m = tf.cast(tf.shape(X)[0], dtype=tf.int32)

        X = X/self.lengthscale+1
        X_l = X + self.lengthscale

        part1 = -2.06637*self.variance**2
        part2 = 0.5*self.lengthscale**2*X*tf.math.erf(0.707107)
        part3 = 0.5*tf.math.square(X_l)*tf.math.erf(0.707107)

        final = part1*(part2+part3)
        #diag = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(part1*(part2+part3))), (-1,))
        diag = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(final)), (-1,))

        return diag

class K_multiplicative(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__()
        self.variance_da = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
        self.lengthscale_da = gpflow.Parameter(10.0, transform=gpflow.utilities.positive())
        self.lengthscale_db = gpflow.Parameter(10.0, transform=gpflow.utilities.positive())

    def K(self, X, X2=None, presliced=None):
        if X2 is None:
            X2 = X
        n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        #X_logscaled = np.zeros((n, 2), np.float64)
        #X2_logscaled = np.zeros((m, 2), np.float64)

        #X = tf.math.log(X+self.lengthscale_da)
        X_logscaled_dim0 = tf.math.log(X[:,0]/self.lengthscale_da+1)
        X_logscaled_dim1 = tf.math.log(X[:,1]/self.lengthscale_db+1)
        X2_logscaled_dim0 = tf.math.log(X2[:,0]/self.lengthscale_da+1)
        X2_logscaled_dim1 = tf.math.log(X2[:,1]/self.lengthscale_db+1)

        X_logscaled = tf.stack([X_logscaled_dim0, X_logscaled_dim1], axis=1)
        X2_logscaled = tf.stack([X2_logscaled_dim0, X2_logscaled_dim1], axis=1)

        X_matrix_da = tf.tile(tf.reshape(X_logscaled[0:n,0],(n,1)), (1,m))
        X_matrix_tr_da = tf.tile(tf.reshape(X2_logscaled[0:m,0], (m,1)), (1,n))
        X_matrix_tr_da = tf.transpose(X_matrix_tr_da)

        X_matrix_db = tf.tile(tf.reshape(X_logscaled[0:n,1],(n,1)), (1,m))
        X_matrix_tr_db = tf.tile(tf.reshape(X2_logscaled[0:m,1],(m,1)), (1,n))
        X_matrix_tr_db = tf.transpose(X_matrix_tr_db)


        diff_X_da = X_matrix_da - X_matrix_tr_da
        diff_X_db = X_matrix_db - X_matrix_tr_db
        #diff_X_dbda = X_matrix_db - X_matrix_tr_da
        #diff_X_dadb = X_matrix_da - X_matrix_tr_db

        K_dada = tf.exp(-0.5*(diff_X_da)**2)
        K_dbdb = tf.exp(-0.5*(diff_X_db)**2)
        #K_dbda = self.variance_da*tf.exp(-0.5*(diff_X_dbda)**2)
        #K_dadb = self.variance_da*tf.exp(-0.5*(diff_X_dadb)**2)

        #return (K_dada * K_dbdb + K_dadb * K_dbda + K_dbda* K_dadb + K_dada * K_dbdb)
        #return (K_dada + K_dbdb + K_dadb + K_dbda + K_dbda + K_dadb + K_dada + K_dbdb)
        #return (K_dada * K_dbdb + K_dadb * K_dbda)
        return self.variance_da * K_dada * K_dbdb

    def K_diag(self, X, presliced=None):
        n = tf.cast(tf.shape(X)[0], dtype=tf.int32)
        m = tf.cast(tf.shape(X)[0], dtype=tf.int32)
        diag = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_da)), (-1,))

        X_logscaled = np.zeros((n, 2), np.float64)
        X2_logscaled = np.zeros((m, 2), np.float64)

        X_logscaled[:,0] = tf.math.log(X[:,0]/self.lengthscale_da+1)
        X_logscaled[:,1] = tf.math.log(X[:,1]/self.lengthscale_db+1)

        diff_X_dbda =  X_logscaled[:,1] -  X_logscaled[:,0]
        diff_X_dadb =  X_logscaled[:,0] -  X_logscaled[:,1]

        K_dbda = self.variance_da*tf.exp(-0.5*(diff_X_dbda)**2)
        K_dadb = self.variance_da*tf.exp(-0.5*(diff_X_dadb)**2)


        #diag_db = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_db)), (-1,))
        #return diag + K_dadb * K_dbda + K_dbda* K_dadb
        #return diag + K_dadb * K_dbda
        return diag #+ K_dadb + K_dbda

class K_multiplicative_time(gpflow.kernels.Kernel):

    def __init__(self):
        super().__init__()
        self.variance_da = gpflow.Parameter(1.0, transform=gpflow.utilities.positive())
        self.lengthscale_da = gpflow.Parameter(10.0, transform=gpflow.utilities.positive())
        self.lengthscale_db = gpflow.Parameter(10.0, transform=gpflow.utilities.positive())
        self.lengthscale_dt = gpflow.Parameter(10.0, transform=gpflow.utilities.positive())

    def K(self, X, X2=None, presliced=None):
        if X2 is None:
            X2 = X
        n=tf.cast(tf.shape(X)[0], dtype=tf.int32)
        m=tf.cast(tf.shape(X2)[0], dtype=tf.int32)

        #X_logscaled = np.zeros((n, 2), np.float64)
        #X2_logscaled = np.zeros((m, 2), np.float64)

        #X = tf.math.log(X+self.lengthscale_da)
        X_logscaled_dim0 = tf.math.log(X[:,0]/self.lengthscale_da+1)
        X_logscaled_dim1 = tf.math.log(X[:,1]/self.lengthscale_db+1)
        X_logscaled_dim2 = tf.math.log(X[:,2]/self.lengthscale_dt+1)
        X2_logscaled_dim0 = tf.math.log(X2[:,0]/self.lengthscale_da+1)
        X2_logscaled_dim1 = tf.math.log(X2[:,1]/self.lengthscale_db+1)
        X2_logscaled_dim2 = tf.math.log(X2[:,2]/self.lengthscale_dt+1)

        X_logscaled = tf.stack([X_logscaled_dim0, X_logscaled_dim1, X_logscaled_dim2], axis=1)
        X2_logscaled = tf.stack([X2_logscaled_dim0, X2_logscaled_dim1, X2_logscaled_dim2], axis=1)

        X_matrix_da = tf.tile(tf.reshape(X_logscaled[0:n,0],(n,1)), (1,m))
        X_matrix_tr_da = tf.tile(tf.reshape(X2_logscaled[0:m,0], (m,1)), (1,n))
        X_matrix_tr_da = tf.transpose(X_matrix_tr_da)

        X_matrix_db = tf.tile(tf.reshape(X_logscaled[0:n,1],(n,1)), (1,m))
        X_matrix_tr_db = tf.tile(tf.reshape(X2_logscaled[0:m,1],(m,1)), (1,n))
        X_matrix_tr_db = tf.transpose(X_matrix_tr_db)

        X_matrix_dt = tf.tile(tf.reshape(X_logscaled[0:n,2],(n,1)), (1,m))
        X_matrix_tr_dt = tf.tile(tf.reshape(X2_logscaled[0:m,2],(m,1)), (1,n))
        X_matrix_tr_dt = tf.transpose(X_matrix_tr_db)


        diff_X_da = X_matrix_da - X_matrix_tr_da
        diff_X_db = X_matrix_db - X_matrix_tr_db
        diff_X_dt = X_matrix_dt - X_matrix_tr_dt

        diff_X_dbda = X_matrix_db - X_matrix_tr_da
        diff_X_dadb = X_matrix_da - X_matrix_tr_db

        K_dada = tf.exp(-0.5*(diff_X_da)**2)
        K_dbdb = tf.exp(-0.5*(diff_X_db)**2)
        K_dtdt = tf.exp(-0.5*(diff_X_dt)**2)


        K_dbda = self.variance_da*tf.exp(-0.5*(diff_X_dbda)**2)
        K_dadb = self.variance_da*tf.exp(-0.5*(diff_X_dadb)**2)

        #return (K_dada * K_dbdb + K_dadb * K_dbda + K_dbda* K_dadb + K_dada * K_dbdb)
        #return (K_dada + K_dbdb + K_dadb + K_dbda + K_dbda + K_dadb + K_dada + K_dbdb)
        #return (K_dada * K_dbdb + K_dadb * K_dbda)
        return self.variance_da * K_dada * K_dbdb * K_dtdt

    def K_diag(self, X, presliced=None):
        n = tf.cast(tf.shape(X)[0], dtype=tf.int32)
        m = tf.cast(tf.shape(X)[0], dtype=tf.int32)
        diag = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_da)), (-1,))

        X_logscaled = np.zeros((n, 3), np.float64)
        X2_logscaled = np.zeros((m, 3), np.float64)

        X_logscaled[:,0] = tf.math.log(X[:,0]/self.lengthscale_da+1)
        X_logscaled[:,1] = tf.math.log(X[:,1]/self.lengthscale_db+1)
        X_logscaled[:,2] = tf.math.log(X[:,2]/self.lengthscale_dt+1)

        diff_X_dbda =  X_logscaled[:,1] -  X_logscaled[:,0]
        diff_X_dadb =  X_logscaled[:,0] -  X_logscaled[:,1]
        #diff_X_dtdt =  X_logscaled[:,0] -  X_logscaled[:,1]

        K_dbda = self.variance_da*tf.exp(-0.5*(diff_X_dbda)**2)
        K_dadb = self.variance_da*tf.exp(-0.5*(diff_X_dadb)**2)


        #diag_db = tf.reshape(tf.fill(tf.stack([n]), tf.squeeze(self.variance_db)), (-1,))
        #return diag + K_dadb * K_dbda + K_dbda* K_dadb
        #return diag + K_dadb * K_dbda
        return diag #+ K_dadb + K_dbda


def likelihood_ratio(llmin, llmax):
    return(2*(llmax-llmin))
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
