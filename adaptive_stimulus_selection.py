'''

This module is used to simulate adaptive stimulus selection.

5 files are saved for each run n_r:

For all n_trials of that run:

- expected value estimates
- map estimates
- frames which were selected
- rods which were selected
- priors 

'''

import torch
import numpy as np
import integration_model as model
from timeit import default_timer as timer
from tqdm import tqdm
from neural_network import get_model


disc = 5
n_trials = 500
n_runs = 10

# to be recovered parameter set [beta_hs, kv, kh, tau, sigma_hp, lambda]
params = [0.0384, 86.2428, 1.4506, 0.8, 0.1134, 0.02]

def stimulus_selection(n_trials, network, inputs, param_values, n_r):
    '''

    n_trials: Number of trials of the experiment.

    network: The feed forward neural network used to compute p(r|x, theta).

    inputs: Parameter sets and stimuli the network computes p(r|x, theta) for. Dimensions = (disc^6 x 162) x 8.
    This matrix also contains lambda in the first column which will not be used as network input.

    param_values: Contains the unique parameter sets from inputs including the lapse rate parmater lambda which is not network input.
    Dimensions = 5^6 x 6

    nr: The number of times the experiment is repeated.

    '''

    n = disc**6
    priors = ((1/disc)**6) * np.ones(n) # assuming all ranges are discretized in the same way

    # Forward passes are split to reduce amount of memory needed. Computed once.

    p_cws = []

    for i in range(0, len(inputs), n):

        # only use beta_hs, kv, kh, tau, sigma_hp x stimuli as network input

        p_cw = network(torch.Tensor(inputs[:, 1:][i:i + n])).detach().numpy().ravel()
        p_cws.append(p_cw)

    # multiply network output with lapse rates

    lapse_rates = inputs[:, 0]
    p_cw_x_theta = np.hstack(p_cws) * (1 - 2 * lapse_rates) + lapse_rates

    theta_f_t = np.zeros(n_trials)
    theta_r_t = np.zeros(n_trials)

    avg_estimated_params = np.zeros((n_trials, 6))
    map_estimated_params = np.zeros((n_trials, 6))
    priors_t = np.zeros((n_trials, n))

    for t in tqdm(range(n_trials)):

        E_entropy_x = np.zeros(162)
        posterior_theta_cw_x = np.zeros(n * 162)
        posterior_theta_ccw_x = np.zeros(n * 162)

        m_likelihood_cw_x = np.sum(p_cw_x_theta[0:n] * priors)
        m_likelihood_ccw_x = 1 - m_likelihood_cw_x

        entropy_cw_x = 0
        entropy_ccw_x = 0

        for j in range(n * 162):  # the total number of parameter stimuli combinations

            if j % n == 0 and j > 0: # compute marginal likelihood once per stimulus

                index = int(j / n)

                E_entropy_x[index-1] = (-entropy_cw_x) * m_likelihood_cw_x + (-entropy_ccw_x) * m_likelihood_ccw_x
                entropy_cw_x = 0
                entropy_ccw_x = 0

                if j < n * 162:

                    m_likelihood_cw_x = np.sum(p_cw_x_theta[index * n:(index+1) * n] * priors)
                    m_likelihood_ccw_x = 1 - m_likelihood_cw_x

            prior = priors[j%n]

            likelihood_cw = p_cw_x_theta[j]
            likelihood_ccw = 1 - likelihood_cw

            post_prob_cw_x = (likelihood_cw * prior) / m_likelihood_cw_x
            post_prob_ccw_x = (likelihood_ccw * prior) / m_likelihood_ccw_x

            posterior_theta_cw_x[j] = post_prob_cw_x
            posterior_theta_ccw_x[j] = post_prob_ccw_x

            entropy_cw_x += post_prob_cw_x * np.log2(post_prob_cw_x) if post_prob_cw_x != 0 else 0
            entropy_ccw_x += post_prob_ccw_x * np.log2(post_prob_ccw_x) if post_prob_ccw_x != 0 else 0

        E_entropy_x[-1] = (-entropy_cw_x) * m_likelihood_cw_x + (-entropy_ccw_x) * m_likelihood_ccw_x

        s_nr = np.argmin(E_entropy_x)

        theta_f = inputs[:, 6][s_nr * n]
        theta_r = inputs[:, 7][s_nr * n]

        theta_f_t[t] = theta_f
        theta_r_t[t] = theta_r

        response = model.simulate_subject_response(theta_f, theta_r, params)

        index_1 = s_nr * n
        index_2 = (s_nr+1) * n

        priors = posterior_theta_cw_x[index_1:index_2] if response else posterior_theta_ccw_x[index_1:index_2]
        avg_estimated_params[t:] = np.sum(param_values * priors[:, None], axis=0)
        map_estimated_params[t:] = param_values[np.argmax(priors)]
        priors_t[t:] = priors

    np.savez_compressed(f'data/adaptive_stimulus_selection/disc_{disc}_avg_estimates_run_{n_r}.npz', avg_estimated_params)
    np.savez_compressed(f'data/adaptive_stimulus_selection/disc_{disc}_map_estimates_run_{n_r}.npz', map_estimated_params)
    np.savez_compressed(f'data/adaptive_stimulus_selection/disc_{disc}_frames_run_{n_r}.npz', theta_f_t)
    np.savez_compressed(f'data/adaptive_stimulus_selection/disc_{disc}_rods_run_{n_r}.npz', theta_r_t)
    np.savez_compressed(f'data/adaptive_stimulus_selection/disc_{disc}_priors_run_{n_r}.npz', priors_t)

if __name__ == '__main__':

    network, _, _ = get_model()
    network.load_state_dict(torch.load('models/relu_100_100/relu_100_100_batch_size_250_fold_1.pt', map_location=torch.device('cpu')))
    network_inputs = np.load(f'data/adaptive_stimulus_selection/disc_{disc}_net_input.npz')['arr_0']
    param_values = np.load(f'data/adaptive_stimulus_selection/disc_{disc}_params.npz')['arr_0']

    times = []
    
    for n_r in range(1,11):

        start = timer()
        stimulus_selection(500, network, network_inputs, param_values, n_r)
        end = timer()
        times.append(end-start)
        
    print(sum(times)/(500*10))
        
    







