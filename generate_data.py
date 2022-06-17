'''  This module is used to generate training data for the networks. '''

import numpy as np
import integration_model as model
from tqdm import tqdm


def generate_data(beta_hs, kappa_v, kappa_h, tau, sigma_hp, rods, frames):

    X = []
    Y = []

    for v1 in tqdm(beta_hs):
        for v2 in kappa_v:
            for v3 in kappa_h:
                for v4 in tau:
                    for v5 in sigma_hp:

                        parameters = [v1, v2, v3, v4, v5]
                        x, y = model.gen_response_probabilities(parameters, rods, frames)
                        X.append(x)
                        Y.append(y)

    X = np.vstack(X)
    Y = np.ravel(Y)

    np.savez_compressed(f'data/training/in_{param_disc}_{rod_disc}_{frame_disc}.npz', X)
    np.savez_compressed(f'data/training/ts_{param_disc}_{rod_disc}_{frame_disc}.npz', Y)

    print('Data generated and saved!')


if __name__ == '__main__':

    # parameter ranges for kappas were calculated from sigmas (v,h) using k = 3994.5 / (sigma^2 + 22.6)

    param_disc = 5
    rod_disc = 20
    frame_disc = 20

    beta_hs = np.linspace(0.0366, 0.1257, param_disc)
    kappa_v = np.linspace(50.6595, 133.6400, param_disc)
    kappa_h = np.linspace(0.5274, 9.9156, param_disc)
    tau = np.linspace(0.79, 1, param_disc)
    sigma_hp = np.linspace(0.0942, 0.2862, param_disc)

    rods = np.linspace(-7, 7, rod_disc)
    frames = np.linspace(-45, 40, frame_disc)

    generate_data(beta_hs, kappa_v, kappa_h, tau, sigma_hp, rods, frames)


