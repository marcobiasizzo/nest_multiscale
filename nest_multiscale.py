import numpy as np
from scipy.integrate import solve_ivp
import time
from contextlib import contextmanager

from marco_nest_utils import utils, visualizer as vsl


class sim_handler:
    def __init__(self, nest_, pop_list_to_ode_, pop_list_to_nest_, ode_params_, sim_time_, sim_period_=1., resolution=0.1,
                 additional_classes=None, CS_stim = [], n_wind = 1):
        self.nest = nest_  # load nest and previously defined spiking n.n.s
        self.pop_list_to_ode = pop_list_to_ode_     # list of neurons pops projecting to mass model
        self.pop_list_to_nest = pop_list_to_nest_  # list of mass models projecting to spiking n.n.s
        self.CS_stim = CS_stim
        self.ode_params = ode_params_  # load odes dictionary params
        # x = - x + S(A*x + B*u)
        # y = C*x
        self.A = ode_params_['A']  # matrix relating mass firing rates
        self.B = ode_params_['B']  # matrix multiplying firing rates inputs
        self.C = ode_params_['C']  # output matrix
        self.lambda_max = ode_params_['lambda_max']  # sigmoid saturation
        self.a = ode_params_['a']  # sigmoid inclination
        self.th = ode_params_['theta']  # sigmoid threshold
        self.q = ode_params_['q']  # const term to be added to the sigmoid

        self.T = sim_period_  # separated simulation time interval, [ms]
        self.T_fin = sim_time_  # total simulation time [ms]
        self.n_wind =n_wind - 1
        self.resolution = resolution    # Nest resolution [ms]

        self.get_system_dimentions()  # derive state, in and out dimensions according to matrices
        self.tau_decay = ode_params_['tau']  # should be provided in ms!
        self.rhs = self.define_ode(ode_params_['const_in'])  # generate the rhs of odes (vectorial)

        self.sd_list = utils.attach_spikedetector(self.nest, self.pop_list_to_ode)

        self.rng = np.random.default_rng(round(time.time() * 1000))
        

        if additional_classes is not None:
            self.additional_classes = additional_classes
            for a_c in self.additional_classes:
                a_c.start(self, sim_time_, sim_period_)
        else:
            self.additional_classes = []


    def get_system_dimentions(self):
        """ Save as class variables the dimensions of state, input and output
            Before doing that, it verifies the coherence of matrix and populations dimensions """

        A_dim = self.A.shape
        assert A_dim[0] == A_dim[1], 'A matrix must be squared'
        self.x_dim = A_dim[0]  # get the number of compartments from the A_matrix rows
        B_dim = self.B.shape
        assert B_dim[0] == self.x_dim, 'B rows should be equal to # of compartments'
        assert B_dim[1] == len(self.pop_list_to_ode), 'Number of inputs pops should be coherent with B clms'
        self.u_dim = B_dim[1]
        C_dim = self.C.shape
        assert C_dim[1] == self.x_dim, 'C clmns should be equal to # of compartments'
        assert C_dim[0] == len((self.pop_list_to_nest))-self.n_wind, 'Number of output pops should be coherent with C raws'
        self.y_dim = C_dim[0]
        print('Loaded matrices!')
        print(f'System dimentions are: d_state={self.x_dim}, d_input={self.u_dim}, d_output={self.y_dim}')


    def define_ode(self, const_in):
        # IMPORTANT!: for odeint x should be a row vector, incoherent with model definition
        # We define x and u as row vectors, so we have to transpose them before matrix products
        if const_in is not None:
            rhs = lambda t, x, u: 1000. / self.tau_decay * (
                        -x + self.lambda_max(x) * self.sigmoid(self.A @ x.T + self.B @ u.T + const_in.T))
        else:
            rhs = lambda t, x, u: 1000. / self.tau_decay * (
                        -x + self.lambda_max(x) * self.sigmoid(self.A @ x.T + self.B @ u.T))
        return rhs

    def sigmoid(self, s):
        return 1. / (1 + np.exp(-self.a * (s - self.th))) + self.q


    def simulate(self, sub_intervals=10, tot_trials=1, pre_sim_time=0.):
        """ Simulate the model made by spiking n.n.s and mass models.
            Total simulation time is self.T_fin, while sampling time is self.T   """

        # define the sub-interval where the ode is solved in discrete time
        sub_interv = sub_intervals  # integrate in 1 T period, sampling every T/sub_interv
        int_t = np.linspace(0, self.T / 1000., sub_interv + 1, endpoint=True)

        # define sigma and window width in ms
        kernel = half_gaussian(sigma=10., width=40., sampling_time=self.T)

        # prepare solution buffers
        u0 = np.array([0.] * self.u_dim)
        u_sol = u0.reshape((1, self.u_dim))     # will contain all solutions over time
        x0 = np.array([0.] * self.x_dim)        # self.ode_params['lambda_max'] / 2  # a typical equilibrium condition
        ode_sol = x0.reshape((1, self.x_dim))   # will contain all states over time
        ode_sol_t = np.array([0.])              # will contain the state time
        y0 = self.C @ x0

        # prepare initial values for calls in for loop
        u = u0
        prev_steps = len(kernel)  # number of steps in the output gaussian filter
        u_old = np.zeros((prev_steps, self.u_dim))  # buffer of zeros for the gaussian filter

        yT = y0
        ode_to_spikes_delay = int(3 / self.T)  # [ms]      # delay before sending inputs to mass models
        yT_buf = np.tile(y0, [ode_to_spikes_delay, 1])  # create a buffer of ode_to_spikes_delay yT

        tt = 0.

        if pre_sim_time > 0.:
            print(f'Starting pre-simulation of {pre_sim_time} ms')
            with self.nest.RunManager():  # allows running consequently nest simulations
                while tt < pre_sim_time:  # run until the lower bound < final time
                    # total elapsed time, also from previous simulations
                    actual_sim_time = tt

                    # 1) Run nest simulation for T ms (in [tt, tt + T]) or (in [actual_sim_time, actual_sim_time + T])
                    # We don't need to pass tt since nest RunManager save actual time value
                    self.nest.Run(self.T)  # with nest run we can evaluate smaller intervals

                    # 2) Transform spikes into f.r.
                    # to be used in the next interval
                    new_u = calculate_instantaneous_fr(self.nest, self.sd_list, self.pop_list_to_ode,
                                                       time=actual_sim_time, T_sample=self.T)

                    # 3) Solve odes for T ms (in [tt, tt + T]), in sub_intervals defined before
                    sol = solve_ivp(self.rhs, t_span=[0, self.T / 1000.], y0=x0, method='RK45', t_eval=int_t, args=(u,))
                    ode_sol = np.concatenate((ode_sol, sol.y[:, 1:].T), axis=0)  # take all the values after initial condition
                    ode_sol_t = np.concatenate((ode_sol_t, sol.t[1:] * 1000. + actual_sim_time), axis=0)
                    xT = sol.y[:, -1]  # keep the final time states

                    # 4) transform fr to spikes
                    # to be used in the next interval
                    yT = self.C @ xT  # calculate the transferred f.r., to be used in next interval
                    yT = yT.reshape(1, self.y_dim)

                    # update values for next integration step:
                    x0 = xT  # update fr state to continue integration
                    # update input values from populations projecting to odes
                    u, u_old = evaluate_fir(u_old, new_u.reshape((1, self.u_dim)), kernel=kernel)
                    if not yT_buf.size and self.n_wind>0:
                        yT_buf = np.ndarray((0,8 + self.n_wind))
                    elif yT_buf.shape[1]==8 and self.n_wind>0:
                        
                        yT_buf = np.insert(yT_buf,1,np.repeat(yT_buf[:,0], self.n_wind) ,axis=1)
                    # yT_tmp = np.tile(yT[:,0].reshape(2,1),2)
                    # yT = np.concatenate((yT_tmp, yT[:,1:]), axis=1)
                    if self.n_wind>0:
                        yT[0] = yT[0]
                        yT = np.insert(yT,1,np.repeat(yT[:,0], self.n_wind),axis=1)
                    # set the future spike trains (in [tt + T, tt + 2T])
                    yT_buf = set_poisson_fr(self.nest, yT, self.pop_list_to_nest, actual_sim_time + self.T,
                                            self.T, self.rng, self.resolution, yT_buf=yT_buf, n_wind = self.n_wind)

                    u_sol = np.concatenate((u_sol, u * np.ones((int_t.shape[0] - 1, self.u_dim))),
                                           axis=0)  # save inputs

                    tt = tt + self.T  # update actual time

        # if trial is not 1, net will be simulated multiple times
        # NOTE THAT network status won't be reset, but just robot status
        for trial in range(tot_trials):
            print(f'Iteration #{trial + 1}')

            for a_c in self.additional_classes:
                a_c.before_loop(self)

            tt = 0.  # used to register simulation time, is the lower bound of the interval

            with self.nest.RunManager():    # allows running consequently nest simulations
                print(f'Starting simulation of {self.T_fin} ms')

                while tt < self.T_fin:      # run until the lower bound < final time

                    # total elapsed time, also from previous simulations
                    actual_sim_time = tt + trial*(self.T_fin) + pre_sim_time

                    # 0) set cortical input
                    #for a_c in self.additional_classes:
                    #    a_c.beginning_loop(self, tt, actual_sim_time)

                    # 1) Run nest simulation for T ms (in [tt, tt + T]) or (in [actual_sim_time, actual_sim_time + T])
                    # We don't need to pass tt since nest RunManager save actual time value
                    self.nest.Run(self.T)  # with nest run we can evaluate smaller intervals

                    # 2) Transform spikes into f.r.
                    # to be used in the next interval
                    new_u = calculate_instantaneous_fr(self.nest, self.sd_list, self.pop_list_to_ode,
                                                       time=actual_sim_time, T_sample=self.T)

                    # 3) Solve odes for T ms (in [tt, tt + T]), in sub_intervals defined before
                    sol = solve_ivp(self.rhs, t_span=[0, self.T / 1000.], y0=x0, method='RK45', t_eval=int_t, args=(u,))
                    ode_sol = np.concatenate((ode_sol, sol.y[:, 1:].T), axis=0)  # take all the values after initial condition
                    ode_sol_t = np.concatenate((ode_sol_t, sol.t[1:] * 1000. + actual_sim_time), axis=0)
                    xT = sol.y[:, -1]  # keep the final time states

                    # 4) transform fr to spikes
                    # to be used in the next interval
                    yT = self.C @ xT  # calculate the transferred f.r., to be used in next interval
                    yT = yT.reshape(1, self.y_dim)

                    # update values for next integration step:
                    x0 = xT     # update fr state to continue integration
                    # update input values from populations projecting to odes
                    u, u_old = evaluate_fir(u_old, new_u.reshape((1, self.u_dim)), kernel=kernel)
                    # set the future spike trains (in [tt + T, tt + 2T])
                    if not yT_buf.size:
                        yT_buf = np.ndarray((0, 8 + self.n_wind))
                    # yT_tmp = np.tile(yT[:,0].reshape(2,1),2)
                    # yT = np.concatenate((yT_tmp, yT[:,1:]), axis=1)
                    if self.n_wind>0:
                        yT[0] = yT[0]
                        rep = np.repeat(yT[:,0], self.n_wind).reshape(1,self.n_wind)
                        yT = np.concatenate([rep,yT],axis = 1)
                    yT_buf = set_poisson_fr(self.nest, yT, self.pop_list_to_nest, actual_sim_time + self.T,
                                                 self.T, self.rng, self.resolution, yT_buf=yT_buf, n_wind = self.n_wind)

                    u_sol = np.concatenate((u_sol, u * np.ones((int_t.shape[0] - 1, self.u_dim))), axis=0)  # save inputs

                    for a_c in self.additional_classes:
                        a_c.ending_loop(self, tt, actual_sim_time)
                        a_c.CS(self,yT, tt, actual_sim_time)

                    tt = tt + self.T  # update actual time

        # save the time evolution of mass model state as self.ode_sol variable
        self.ode_sol = ode_sol
        self.ode_sol_t = ode_sol_t
        self.u_sol = u_sol


def set_poisson_fr(nest_, fr, target_pop, time, T_sample, random_gen, resolution, yT_buf=None, sin_weight=1., in_spikes = "active", n_wind = 0):
    """ Set the firing rate for a list of poisson generators (which are pops of neurons) """
    # first, save yT in yT_buf
    if yT_buf is not None:
        yT_buf = np.concatenate((yT_buf, fr), axis=0)
        set_yT = yT_buf[0, :]
        yT_buf = yT_buf[1:, :]    # discard the old elem, which we will set now
    else:
        set_yT = np.tile(fr, [n_wind])

    # bkgroung_fr = np.array([0, 162.5, 162.5, 486., 486., 642.6, 642.6, 700.4])

    for idx, poiss in enumerate(target_pop):
        if set_yT[idx] < 0.:        # solving numerical problem generating negative fr
            # # print(set_yT[idx])
            set_yT[idx] = 0.
        # if in_spikes == "EBCC":
        #     resto = time%T_sample
        #     start = time-resto
        #     stop = time-resto+T_sample
        #     generator_params = {"start": start, "stop": stop, "rate":fr[0]}
        # elif in_spikes == "active":
        factor = [7000/n_wind for i in range(n_wind)]
        factor = [n_wind for i in range(n_wind)]

        factor.extend([1 for i in range(8)])
        spike_times = generate_poisson_trains(poiss, set_yT[idx]*factor[idx], T_sample, time, random_gen, resolution)  # long as number of neurons in pop
        # spike_times = self.generate_poisson_trains(poiss, set_yT[idx] + bkgroung_fr[idx], self.T, time)  # long as number of neurons in pop
        generator_params = [{"spike_times": s_t, "spike_weights": [sin_weight] * len(s_t)} for s_t in spike_times]
            
        nest_.SetStatus(poiss, generator_params)

    return yT_buf

def generate_poisson_trains_ebcc(poisson, fr, T, time, rand_gen, resolution):
    resto = time%T
    start = time-resto
    stop = time-resto+T
    generator_params = {"start": start, "stop": stop, "rate":fr[0]}

    return poisson, generator_params

def generate_poisson_trains(poisson, fr, T, time, rand_gen, resolution):
    """ Generate a train of spikes as a Poisson process
        In this version we evaluate the number of spikes in
        1 interval, and we place them using linspace        """
    if isinstance(poisson, int):
        l = 1
    else:
        l = len(poisson)  # number of neurons in the spike generator population

    # evaluate the spikes to be inserted in 1 interval (one for each neuron -> vector of draws)
    # occurrences = np.random.poisson((fr + 4) / 1000 * T, size=l)   # sample from a poisson distribution

    occurrences = rand_gen.poisson(fr / 1000 * T, size=l)  # will be summed with BG noise
    # print(occurrences)
    # occurrences = np.random.poisson(fr, size=l)  # sample from a poisson distribution

    method = 'uniform'
    # generates a spike train with the spikes "at the centre" of the interval + gaussian noise
    if method == 'gaussian':
        gaussian_sd = lambda occ_: T / (occ_ * 4)  # define sd as function of T and occ_, 4 is experimental
        spike_trains = [  # create a list of spike trains, one for each spike generator
            np.sort(  # sort it by spiking time (should be already sorted)
                time +  # sum the time of the beginning of the interval
                np.round(  # approximate to 0.1 ms
                    1./resolution *  # necessary to produce spike times coherent with nest resolution
                    np.linspace(T / (occ * 2), T * (1. - 1. / (occ * 2)), num=occ,
                                endpoint=True)  # place at the centre
                    + rand_gen.normal(0, gaussian_sd(occ), size=occ),  # add gaussian noise
                    0)) * resolution  # round definition at 0.1 ms
            if occ != 0 else [] for occ in occurrences]  # replace with empty list if no occurrences to be places

    if method == 'uniform':
        spike_trains = []
        for occ in occurrences:
            if occ == 0:
                train = []
                spike_trains += [train]
            elif occ > 0:
                # init_interval_values = time + np.linspace(0, T * (1. - 1. / occ), num=occ, endpoint=True)  #
                init_interval_values = time + np.linspace(0 + 0.06, T - 0.06, num=occ, endpoint=False)  #
                train = init_interval_values + rand_gen.uniform(0, (T - 0.12) / occ, size=occ)
                # print(np.round(train, 1))
                spike_trains += [np.round(1./resolution * train, 0) * resolution]

    # check if spikes are outside the interval
    for s_t, occ in zip(spike_trains, occurrences):
        if len(s_t) > 0:  # if not empty
            # print(f'pre = {s_t}')
            # maintain the spikes strictly inside the interval
            # s_t[s_t < time + 0.1] = time + 0.1
            # s_t[s_t < time] = time
            # s_t[s_t > time + T - 0.1] = time + T - 0.1
            # s_t[s_t > time + T] = time + T

            # if two spikes are contemporary, move one forward
            if len(s_t) > 1:
                while any(np.diff(s_t)) == 0:  # if any couple of spike times is equal
                    d = np.concatenate((-1., np.diff(s_t)),
                                       axis=None)  # has elem=0 if same spike times. -1 is useful to maintain the dimention
                    s_t[d == 0] += resolution  # move the second element forward
                    # print(s_t)

            s_t.sort()
            # print(f'post = {s_t}')
        assert len(s_t) == occ, 'Some spikes are lost!'
        # assert all(s_t > time) and all(s_t < time + T), 'spikes outside the interval!'

    return spike_trains


def evaluate_fir(old_u, new_u, kernel, steps_from_begin=None):
    """ Calculate the new input to ode
        considering the #memeory elements before """
    concat = np.concatenate((old_u, new_u), axis=0)  # concatenate along the time axis
    old_u = concat[1:, :]  # update old_u discarding the first value and using u_new
    # u = concat.mean(axis=0)  # new u is the average of new and old values
    u = np.dot(old_u.T, kernel)
    # if steps_from_begin is not None:
    #     if steps_from_begin < len(kernel):
    #         u = u * len(kernel) / steps_from_begin
    return u, old_u


def calculate_instantaneous_fr(nest_, sd_list, pop_list, time, T_sample):
    fr_list = []

    for sd, dim_pop in zip(sd_list, utils.get_pop_dim(pop_list)):
        spikes = nest_.GetStatus(sd, "events")[0]["times"]
        new_spikes = spikes > time
        fr = sum(new_spikes) / (T_sample / 1000)
        # fr = sum(new_spikes) / (dim_pop)
        fr = fr / dim_pop

        fr_list = fr_list + [fr]
    return np.array(fr_list)

def calculate_instantaneous_burst_activity(nest_, sd_list, pop_list, time, t_prev):
    burst_list = []
    BURST_TIME = 5

    for sd, pop, t_prev_i in zip(sd_list, pop_list, t_prev):
        dim_pop = len(pop)
        min_id = min(pop)

        ISI_list = [0 for _ in range(dim_pop*2)]  # list of list, will contain the ISI for each neuron
        for tt, idx in zip(nest_.GetStatus(sd, "events")[0]["times"], nest_.GetStatus(sd, "events")[0]["senders"] - min_id ):
            if tt > time:  # consider just element after t_start
                if t_prev_i[idx] == -1:  # first spike of the neuron
                    t_prev_i[idx] = tt
                else:
                    ISI = (tt - t_prev_i[idx])  # inter spike interval
                    if ISI != 0:
                        if ISI < BURST_TIME:
                            ISI_list[idx] = ISI_list[idx] + 1
                        t_prev_i[idx] = tt  # update the last spike time

        burst_list = burst_list + [sum(ISI_list)]
    # if burst_list != [0, 0]:
    #     print(burst_list)
    return np.array(burst_list), t_prev


def generate_ode_dictionary(A_matrix, B_matrix, C_matrix, theta_vec, lambda_max_vec, a_vec, q_vec, tau_val,
                            const_in=None):
    return {'A': A_matrix,
            'B': B_matrix,
            'C': C_matrix,
            'theta': theta_vec,
            'lambda_max': lambda_max_vec,
            'a': a_vec,
            'q': q_vec,
            'tau': tau_val,
            'const_in': const_in}

def generate_ode_empty_dictionary():
    return {'A': np.zeros([1, 1]),
            'B': np.zeros([1, 1]),
            'C': np.zeros([1, 1]),
            'theta': np.ones(1),
            'lambda_max': lambda x: np.ones(1),
            'a': np.ones(1),
            'q': np.ones(1),
            'tau': 10.,
            'const_in': None}

    # def generate_poisson_trains(self, spike_generator, fr, time, last_sp):
    #     """ Generate a train of spikes as a Poisson process
    #         Here we try to draw them from an exponential"""
    #     spike_trains = []
    #     for _, last_t in zip(spike_generator, last_sp):  # for every spike generator directed to a single area
    #         sp_train = []
    #         drawn_time = np.round(np.random.exponential(1000. / fr), 1)[0]
    #         j = last_t + drawn_time
    #         while j < time + self.T:
    #             sp_train = sp_train + [j if j > time else time + 0.1]
    #             drawn_time = np.round(np.random.exponential(1000. / fr), 1)[0]
    #             j = sp_train[-1] + drawn_time
    #             last_t = sp_train[-1]
    #         spike_trains = spike_trains + [sp_train]
    #     return spike_trains


def half_gaussian(sigma, width, sampling_time):
    s = sigma/sampling_time
    w = int(width/sampling_time)

    time_p = w - np.linspace(0, w, w+1)   # create kernel linspace
    # create a half gaussian kernel:
    kernel = 1. / (np.sqrt(2. * np.pi) * s) * np.exp(-np.power((time_p - 0.) / s, 2.) / 2)
    kernel = kernel/kernel.sum()                # normalize
    # vsl.simple_plot(-time_p, kernel)          # visualize

    return kernel