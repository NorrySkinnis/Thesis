''' This modules is used to generate response probabilities from the Bayesian model.'''

import numpy as np
from scipy.stats import vonmises, norm
from scipy.interpolate import interp1d

x_degrees = np.linspace(-200, 200, 10000)
x_radians = np.radians(x_degrees)

def h_space_prior(sigma_hp):

    hs_estimate = norm.pdf(x_radians, loc=0, scale=sigma_hp)

    return hs_estimate

def vest_likelihood(beta_hs):

    hs_estimate = norm.pdf(x_radians, loc=0, scale=beta_hs)

    return hs_estimate

def cont_likelihood(theta_frame, kappa_v, kappa_h, tau):

    k1 = kappa_v - (1 - np.cos(np.abs(2 * (-theta_frame)))) * tau * (kappa_v - kappa_h)
    k2 = kappa_h + (1 - np.cos(np.abs(2 * (-theta_frame)))) * (1 - tau) * (kappa_v - kappa_h)

    k = [k1, k2, k1, k2]
    phi = np.radians([0, 90, 180, 270])

    hs_estimate = sum([vonmises.pdf(phi[i] + theta_frame - x_radians, k[i]) for i in range(4)])

    return hs_estimate

def h_space_posterior(theta_frame, params):

    beta_hs = params[0]
    kappa_v = params[1]
    kappa_h = params[2]
    tau = params[3]
    sigma_hp = params[4]

    hs_estimate_pdf = cont_likelihood(theta_frame, kappa_v, kappa_h, tau) * \
                      vest_likelihood(beta_hs) * \
                      h_space_prior(sigma_hp)

    hs_estimate_pdf = hs_estimate_pdf / sum(hs_estimate_pdf)
    hs_estimate_cdf = np.cumsum(hs_estimate_pdf)

    hs_estimate_pdf = interp1d(x_degrees, hs_estimate_pdf)
    p_cw_responses = interp1d(x_degrees, 1 - hs_estimate_cdf)

    return p_cw_responses, hs_estimate_pdf

def gen_response_probabilities(params, rods, frames):
    '''
    Used from generate_data.py to generate input and targets for the networks of a single parameter setting.

    params: List of parameters which has be in the order: beta_hs, k_v, k_h, tau, sigma_hp
    rods: List of rod orientations used as stimuli.
    frames: List of frame orientations used as stimuli. '''

    n_rows = len(rods) * len(frames)
    X = np.zeros((n_rows, len(params) + 2))
    y = np.zeros(n_rows)

    for i, theta_f in enumerate(frames):

        p_cw_responses, _ = h_space_posterior(np.radians(theta_f), params)

        for j, theta_r in enumerate(rods):

            X[i * (len(rods)) + j] = np.concatenate((params, [theta_f, theta_r]), axis=0)
            y[i * (len(rods)) + j] = p_cw_responses(-theta_r)

    return X, y

def simulate_subject_response(theta_f, theta_r, params):
    '''
    Used from adaptive_stimulus_selection.py to simulate a subject response for a specific stimulus.

    theta_f: Frame orientation of the stimulus.
    theta_r: Rod orientation of the stimulus.
    params: [beta_hs, kv, kh, tau, sigma_hp, lambda]. '''

    lapse_rate = params[-1]

    p_cw_responses, _ = h_space_posterior(np.radians(theta_f), params)
    p_cw_response = p_cw_responses(-theta_r)
    response = np.random.binomial(1, lapse_rate + (1 - 2 * lapse_rate) * p_cw_response)

    return response