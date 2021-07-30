import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm
from synergy.combination import MuSyC # or BRAID, Zimmer
from synergy.combination import Loewe, Bliss # or Bliss, ZIP, HSA, Schindler, CombinationIndex
import pandas as pd
import matplotlib
import scipy
from matplotlib import pyplot as plt
from synergy.utils import plots

from synergy.utils.dose_tools import grid
import numpy as np

from synergy.single import Hill


# synthetic Greco data

df = pd.read_csv('GrecoSimulatedData.csv', sep=';')

df = df.sort_values(by=['Dose1','Dose2'])

df = df.iloc[2:]
df['Response'].iloc[0] = 106.7

print(df)

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

data = pd.concat([pd.DataFrame(Dose_AB), pd.DataFrame(Effect)], axis=1)
data.columns = ['Dose_A', 'Dose_B', 'Effect']

X1 = data[data['Dose_B']==0]['Dose_A'].to_numpy().reshape(-1,1).astype(float)
Y1 = data[data['Dose_B']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

X2 = data[data['Dose_A']==0]['Dose_B'].to_numpy().reshape(-1,1).astype(float)
Y2 = data[data['Dose_A']==0]['Effect'].to_numpy().reshape(-1,1).astype(float)

Dose_A = data[data['Dose_B']==0]['Dose_A'].to_numpy().reshape(-1,1).astype(float).flatten()
Effect_A = data[data['Dose_B']==0]['Effect'].to_numpy().reshape(-1,1).astype(float).flatten()

Dose_B = data[data['Dose_A']==0]['Dose_B'].to_numpy().reshape(-1,1).astype(float).flatten()
Effect_B = data[data['Dose_A']==0]['Effect'].to_numpy().reshape(-1,1).astype(float).flatten()


df_a = pd.concat([pd.DataFrame(Dose_A), pd.DataFrame(Effect_A)], axis=1)
df_a.columns = ['x', 'y']

df_b = pd.concat([pd.DataFrame(Dose_B), pd.DataFrame(Effect_B)], axis=1)
df_b.columns = ['x', 'y']

n_points_hill = 6


# resample with replacement each row
boot_slopes = []
boot_interc = []

boot_E0 = []
boot_Emax = []
boot_h = []
boot_C = []


n_boots = 100

d = np.linspace(np.min(Dose_A),np.max(Dose_A))
y_pred_array = np.zeros((n_boots, len(d)))

#plt.figure()
for _ in range(n_boots):
 # sample the rows, same size, with replacement
 sample_df = df_a.sample(n=n_points_hill, replace=True)
 print(sample_df)
 # fit a linear regression

 E0 = 1
 Emax = 0
 h = 2.3
 C = 1e-2

 model = Hill(E0, Emax, h, C)

 model.fit(sample_df['x'], sample_df['y'])

 d = np.linspace(np.min(Dose_A),np.max(Dose_A))
 model.E(d)
 parameters = model.get_parameters()
 boot_E0.append(parameters[0])
 boot_Emax.append(parameters[1])
 boot_h.append(parameters[2])
 boot_C.append(parameters[3])

 # plot a greyed out line
 y_pred =model.E(df_a['x'])
 y_pred_temp =model.E(d)
 y_pred_array[_,:] = y_pred_temp.flatten()
 plt.plot(d, y_pred_temp, color='purple', alpha=0.2)
# add data points
plt.scatter(Dose_A, Effect_A, color='purple')
plt.plot(Dose_A, y_pred, linewidth=2, color='purple')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs y')
plt.show()

d = np.linspace(np.min(Dose_A),np.max(Dose_A))
boot_lower_bound = np.zeros(10)
boot_upper_bound = np.zeros(10)
n_d = len(d)-1

for _ in range(10):
    boot_lower_bound[_] = np.min(y_pred_array[_,:])
    boot_upper_bound[_] = np.max(y_pred_array[_,:])


print(boot_lower_bound)
print(boot_upper_bound)
