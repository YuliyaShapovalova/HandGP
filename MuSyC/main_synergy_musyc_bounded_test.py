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

'''''
Define and optimize the models
'''''


'''''
Define and optimize the models without bounds
'''''
model_without_bounds = MuSyC()
model_without_bounds.fit(df['drug1.conc'], df['drug2.conc'], df['effect'],bootstrap_iterations=100)
model_without_bounds.get_parameters(confidence_interval=95)

ci_without_bounds = model_without_bounds.get_parameters(confidence_interval=95)

'''''
Define and optimize the models with bounds
'''''

model_with_bounds = MuSyC(E1_bounds=(0.0,100.0),E2_bounds=(0.0,100.0), E3_bounds=(0.0,100.0))
model_with_bounds.fit(df['drug1.conc'], df['drug2.conc'], df['effect'],bootstrap_iterations=100)
model_with_bounds.get_parameters(confidence_interval=95)

ci_with_bounds = model_with_bounds.get_parameters(confidence_interval=95)

'''''
Define and optimize the models with bounds, number of samples 1000
'''''

model_with_bounds_bootsrtap1000 = MuSyC(E1_bounds=(0.0,100.0),E2_bounds=(0.0,100.0), E3_bounds=(0.0,100.0))
model_with_bounds_bootsrtap1000.fit(df['drug1.conc'], df['drug2.conc'], df['effect'],bootstrap_iterations=1000)
model_with_bounds_bootsrtap1000.get_parameters(confidence_interval=95)

ci_with_bounds_bootsrtap1000 = model_with_bounds_bootsrtap1000.get_parameters(confidence_interval=95)

print('Without bounds', ci_without_bounds)
print('With bounds', ci_with_bounds)
print('With bounds, bootstrap_samples = 1000', ci_with_bounds_bootsrtap1000)
