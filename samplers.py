import numpy as np

def random_walk_metropolis(log_prob, theta0, step, n_steps, rng):
    discard_rate = 0.1 # we generate more samples than n_steps in order to discard the first fraction of samples before burn-in
    n_total = int(n_steps / (1-discard_rate))
    x = np.zeros((n_total,theta0.size))
    x[0] = theta0
    for i in range(n_total-1):
        u = rng.random()
        z = x[i] + step * rng.normal(size=x[i].shape)
        log_a = min(0, log_prob(z)-log_prob(x[i]))
        a = np.exp(log_a)
        if u < a:
            x[i+1] = z
        else:
            x[i+1] = x[i]
    return x[-n_steps:]

def mala(log_prob, grad_log_prob, theta0, step, n_steps, rng):
    discard_rate = 0.1 # we generate more samples than n_steps in order to discard the first fraction of samples before burn-in
    n_total = int(n_steps / (1-discard_rate))
    x = np.zeros((n_total,theta0.size))
    x[0] = theta0
    for i in range(n_total-1):
        u = rng.random()
        z = x[i] + 0.5*step**2 * grad_log_prob(x[i]) + step * rng.normal(size=x[i].shape)
        q_x_to_z = np.sum(-((z-x[i] -0.5*step**2*grad_log_prob(x[i]))**2)/(2*step**2))
        q_z_to_x = np.sum(-((x[i]-z -0.5*step**2*grad_log_prob(z))**2)/(2*step**2))
        a = min(1, np.exp(log_prob(z)-log_prob(x[i])+ q_z_to_x - q_x_to_z))
        if u < a:
            x[i+1] = z
        else:
            x[i+1] = x[i]
    return x[-n_steps:]

def hmc(log_prob, grad_log_prob, theta0, eps, L, n_steps, rng, show_energy_error=False):
    discard_rate = 0.1 # we generate more samples than n_steps in order to discard the first fraction of samples before burn-in
    n_total = int(n_steps / (1-discard_rate))
    dim = theta0.size
    x = np.zeros((n_total,dim * 2))
    x[0,:dim] = theta0
    M = np.eye(dim) # M is identity
    energy_error = np.zeros(n_total-1)
    for i in range(n_total-1):
        p_new = np.matmul(M,rng.normal(size=theta0.shape))
        q_init = x[i,:dim]

        #### computing psi(q^(n),p_new)
        q_cur = q_init
        p_cur = p_new
        for j in range(L):
            q_half = q_cur +(eps/2)* np.linalg.solve(M,p_cur)
            p_next = p_cur + eps*grad_log_prob(q_half)
            q_next = q_half +(eps/2)* np.linalg.solve(M,p_next)
            q_cur = q_next
            p_cur = p_next
        proposal = np.concatenate((q_cur,p_cur))
        #### proposal = psi(q^(n),p_new)
        
        H_x = 0.5* (p_new @ np.linalg.solve(M,p_new)) - log_prob(q_init)
        H_z = 0.5* (p_cur @ np.linalg.solve(M,p_cur)) - log_prob(q_cur)
        a = min(1, np.exp(H_x-H_z))
        energy_error[i] = (H_x-H_z)
        u = rng.random()
        if u < a:
            x[i+1] = proposal
        else:
            x[i+1] = np.concatenate((q_init, -p_new))
    if show_energy_error:
        print(f'The median energy error is: {np.median(energy_error)}')
    return x[-n_steps:,:dim]

def acceptance_rate(x):
    n_accepted = 0
    n_total = len(x)
    for i in range(len(x)-1):
        if not np.array_equal(x[i+1],x[i]):
            n_accepted += 1
    return n_accepted/n_total

def r_hat(chains):
    n_chains, n_samples, n_dim = chains.shape
    means = np.mean(chains, axis=1)
    total_mean = np.mean(means, axis=0)
    B = (n_samples/(n_chains-1)) * np.sum((means - total_mean)**2,axis=0)
    W_times_M = 0
    for m in range(n_chains):
        sample_var_chain = (1/(n_samples - 1)) * np.sum((chains[m] - means[m])**2,axis=0)
        W_times_M += sample_var_chain
    W = W_times_M/n_chains
    r = ((((n_samples-1)/n_samples) * W) + (1/n_samples) * B) / W
    r_hat = np.sqrt(r)
    return r_hat

def effective_sample_size(x):
    N = len(x)
    sample_mean = np.mean(x,axis=0)
    x_tilde = x - sample_mean
    x_tilde = np.append(x_tilde, np.zeros(N))
    xi = np.fft.fft(np.append(x_tilde, np.zeros(N)))
    gamma = np.fft.ifft(np.abs(xi)**2)
    rho = []
    for k in range(N):
        rho.append(gamma[k] / gamma[0])
        if rho[k-1] + rho[k] < 0:
            k_star = k-1
            break
        elif k == N-1:
            return np.nan
    sum = 0
    for k in range(1,k_star):
        sum += (1-((k)/N)) * rho[k]
    tau = 1 + 2 * sum.real
    return N / tau