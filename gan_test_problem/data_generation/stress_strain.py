import numpy as np
from tqdm import tqdm

def get_stress(strains, E, s_y, H):
    e_y = s_y / E
    elastic_strains = strains.copy()
    elastic_strains[elastic_strains > e_y] = e_y
    plastic_strains = strains - elastic_strains
    stresses = elastic_strains*E + plastic_strains*H
    return stresses

def generate_samples(max_strain, n_strain, n_samples):
    strain = np.linspace(0, max_strain, n_strain + 1)[1:]
    stresses = np.empty((n_samples, n_strain))
    for i in tqdm(range(n_samples), desc='Generating samples'):
        E = np.random.normal(1000, 50)
        s_y = np.random.normal(10, 0.5)
        H = np.random.normal(50, 5)
        stresses[i] = get_stress(strain, E, s_y, H)
    return stresses, strain

def generate_stress_samples(max_strain, num_strains, n_samples, preprocessing):
    stresses, _ = generate_samples(max_strain, num_strains, n_samples)
    stresses = np.array(stresses)
    
    scaled_stresses, stress_scaler = preprocessing(stresses)
    
    return scaled_stresses, stress_scaler