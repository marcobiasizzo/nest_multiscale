import numpy as np
from scipy.integrate import odeint
import time
from contextlib import contextmanager

from marco_nest_utils import utils, visualizer as vsl


class sim_handler:
    def __init__(self, nest_, pop_list_to_ode_, pop_list_to_nest_, ode_params_, sim_time_, sim_period_=1,
                 cereb_control_class_=None, robot_=None):
        self.nest = nest_  # load nest and previously defined spiking n.n.s
        self.pop_list_to_ode = pop_list_to_ode_     # list of neurons pops projecting to mass model
        self.pop_list_to_nest = pop_list_to_nest_  # list of mass models projecting to spiking n.n.s

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

        self.get_system_dimentions()  # derive state, in and out dimensions according to matrices
        self.tau_decay = ode_params_['tau']  # should be provided in ms!
        self.rhs = self.define_ode(ode_params_['const_in'])  # generate the rhs of odes (vectorial)

        self.sd_list = utils.attach_spikedetector(self.nest, self.pop_list_to_ode)

        self.rng = np.random.default_rng(round(time.time() * 1000))

        self.robot = robot_
        if robot_ is not None:
            assert cereb_control_class_ is not None, 'You should provide also the robot cereb_control class list'
            self.robot.set_robot_T(sim_period_)
            self.n_joints = len(self.robot.q)
            assert self.n_joints == len(cereb_control_class_.des_trajec), 'Should provide as many trajectory as robot joints'

        self.cereb_control_class = cereb_control_class_
        if cereb_control_class_ is not None:
            assert robot_ is not None, 'You should provide also the robot class'
            self.pop_list_to_robot = self.cereb_control_setup(cereb_control_class_)
            self.sd_list_R = utils.attach_spikedetector(self.nest, self.pop_list_to_robot)



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
        assert C_dim[0] == len((self.pop_list_to_nest)), 'Number of output pops should be coherent with C raws'
        self.y_dim = C_dim[0]
        print('Loaded matrices!')
        print(f'System dimentions are: d_state={self.x_dim}, d_input={self.u_dim}, d_output={self.y_dim}')


    def define_ode(self, const_in):
        # IMPORTANT!: for odeint x should be a row vector, incoherent with model definition
        # We define x and u as row vectors, so we have to transpose them before matrix products
        if const_in is not None:
            rhs = lambda x, t, u: 1000. / self.tau_decay * (
                        -x + self.lambda_max(x) * self.sigmoid(self.A @ x.T + self.B @ u.T + const_in.T))
        else:
            rhs = lambda x, t, u: 1000. / self.tau_decay * (
                        -x + self.lambda_max(x) * self.sigmoid(self.A @ x.T + self.B @ u.T))
        return rhs

    def sigmoid(self, s):
        return 1. / (1 + np.exp(-self.a * (s - self.th))) + self.q


    def cereb_control_setup(self, cereb_control_class_):
        # set proper sampling time in cereb control
        self.cereb_control_class.set_T_and_max_func(self.T, self.T_fin)

        # divide dcn in pos tau and neg tau
        pop_list_to_robot = self.cereb_control_class.Cereb_class.get_dcn_indexes(self.n_joints)

        self.u_R_dim = len(pop_list_to_robot)   # total number of moments
        self.y_R_dim = self.u_R_dim     # total number of error signals to be sent to IO

        return pop_list_to_robot    # , pf_pc_conn, w0


    def simulate(self, sub_intervals=10, tot_trials=1, modified_tau_=None, cortical_input=None, healthy_cortical_input=None):
        """ Simulate the model made by spiking n.n.s and mass models.
            Total simulation time is self.T_fin, while sampling time is self.T   """

        # define the sub-interval where the ode is solved in discrete time
        sub_interv = sub_intervals  # integrate in 1 T period, sampling every T/sub_interv
        int_t = np.linspace(0, self.T / 1000., sub_interv + 1, endpoint=True)

        # define sigma and window width in ms
        kernel = half_gaussian(sigma=10., width=40., sampling_time=self.T)
        kernel_robot = half_gaussian(sigma=600., width=300., sampling_time=self.T)

        # prepare solution buffers
        u0 = np.array([0.] * self.u_dim)
        u_sol = u0.reshape((1, self.u_dim))  # will contain all solutions over time
        x0 = np.array([0.] * self.x_dim)  # self.ode_params['lambda_max'] / 2  # a typical equilibrium condition
        ode_sol = x0.reshape((1, self.x_dim))  # will contain all states over time
        y0 = self.C @ x0

        # prepare initial values for calls in for loop
        u = u0
        prev_steps = len(kernel)  # number of steps in the output gaussian filter
        u_old = np.zeros((prev_steps, self.u_dim))  # buffer of zeros for the gaussian filter

        yT = y0
        ode_to_spikes_delay = int(3 / self.T)  # [ms]      # delay before sending inputs to mass models
        yT_buf = np.tile(y0, [ode_to_spikes_delay, 1])  # create a buffer of ode_to_spikes_delay yT

        if self.cereb_control_class is not None:
            tau0 = np.array([0.] * self.u_R_dim)
            tau_sol = None  # will contain all solutions over time

            tau = tau0
            prev_steps = len(kernel_robot)
            tau_old = np.zeros((prev_steps, 2*self.n_joints))

            sd_list_io, list_io = self.attach_io_spike_det()

            io0 = np.array([0.] * 2)
            io_sol = None  # will contain all solutions over time

            io = io0
            prev_steps_io = len(kernel)
            io_old = np.zeros((prev_steps, 2*self.n_joints))

            self.robot.reset_state(self.cereb_control_class.starting_state)

            e_old = np.zeros((int(140 / self.T), self.n_joints))

        # if trial is not 1, net will be simulated multiple times
        # NOTE THAT network status won't be reset, but just robot status
        for trial in range(tot_trials):
            print(f'Iteration #{trial + 1}')

            if self.cereb_control_class is not None:
                self.robot.reset_state(self.cereb_control_class.starting_state)
                tau_old = np.zeros((prev_steps, 2 * self.n_joints))
                io_old = np.zeros((prev_steps_io, 2 * self.n_joints))
                q_val = [0.]  # visualizer buffer

            tt = 0.  # used to register simulation time, is the lower bound of the interval

            with self.nest.RunManager():    # allows running consequently nest simulations
                print(f'Starting robot simulation of {self.T_fin} ms')

                while tt < self.T_fin:      # run until the lower bound < final time

                    # total elapsed time, also from previous simulations
                    actual_sim_time = tt + trial*(self.T_fin)

                    # 0) set cortical input
                    if self.cereb_control_class is not None:
                        # set future RBF input
                        if cortical_input is None:
                            self.cereb_control_class.generate_RBF_activity(trajectory_time=tt, simulation_time=actual_sim_time)
                        else:
                            if healthy_cortical_input is not None and trial >= 20:
                                self.cereb_control_class.generate_RBF_activity(trajectory_time=tt,
                                                                               simulation_time=actual_sim_time,
                                                                               ctx_input=healthy_cortical_input[int(tt)] * 200.)
                            else:
                                self.cereb_control_class.generate_RBF_activity(trajectory_time=tt, simulation_time=actual_sim_time,
                                                      ctx_input=cortical_input[int(tt)] * 200.)

                    # 1) Run nest simulation for T ms (in [tt, tt + T]) or (in [actual_sim_time, actual_sim_time + T])
                    # We don't need to pass tt since nest RunManager save actual time value
                    self.nest.Run(self.T)  # with nest run we can evaluate smaller intervals

                    # 2) Transform spikes into f.r.
                    # to be used in the next interval
                    new_u = calculate_instantaneous_fr(self.nest, self.sd_list, self.pop_list_to_ode,
                                                       time=actual_sim_time, T_sample=self.T)

                    # 3) Solve odes for T ms (in [tt, tt + T]), in sub_intervals defined before
                    sol = odeint(self.rhs, x0, int_t, args=(u,))
                    ode_sol = np.concatenate((ode_sol, sol[1:]), axis=0)  # take all the values after initial condition
                    xT = sol[-1, :]  # keep the final time states

                    # 4) transform fr to spikes
                    # to be used in the next interval
                    yT = self.C @ xT  # calculate the transferred f.r., to be used in next interval
                    yT = yT.reshape(1, self.y_dim)

                    # update values for next integration step:
                    x0 = xT     # update fr state to continue integration
                    # update input values from populations projecting to odes
                    u, u_old = evaluate_fir(u_old, new_u.reshape((1, self.u_dim)), kernel=kernel)
                    # set the future spike trains (in [tt + T, tt + 2T])
                    yT_buf = self.set_poisson_fr(yT, self.pop_list_to_nest, actual_sim_time + self.T, yT_buf)

                    u_sol = np.concatenate((u_sol, u * np.ones((int_t.shape[0] - 1, self.u_dim))), axis=0)  # save inputs

                    if self.cereb_control_class is not None:
                        # update io value for next step
                        # new_io = calculate_instantaneous_fr(self.nest, self.sd_list_IO, self.pop_list_from_robot,
                        #                                     time=actual_sim_time, T_sample=self.T)
                        # io, io_old = evaluate_fir(io_old, new_io.reshape((1, 2)), kernel=kernel)
                        # # steps_from_begin=int(tt/self.T)+1)
                        # io_sol = np.concatenate((io_sol, io * np.ones((1, 2))), axis=0)

                        # a) Update robot position
                        # use tau of (previous step!) to update robot position
                        tau_pos = tau[1::2]    # to take odd elements:   [1::2]
                        tau_neg = tau[0::2]    # to take even elements:  [0::2]

                        if self.u_dim == 1:     # TO BE improved
                            if self.n_joints == 1: k = 0.4
                            if self.n_joints == 2: k = 0.5
                            if modified_tau_ is not None:
                                if cortical_input is None:
                                    self.robot.update_state(
                                        (tau_pos - tau_neg) * k + modified_tau_(tt) * 1)
                                else:
                                    self.robot.update_state(
                                        (tau_pos - tau_neg) * k + modified_tau_(tt) * cortical_input[int(tt)] / 0.41019194941123904)
                            else:
                                # self.robot.update_state(2)
                                self.robot.update_state((tau_pos - tau_neg)[0] * k)
                        else:
                            # self.robot.update_state((tau_pos - tau_neg)*0.5 + 8 * xT[0])
                            self.robot.update_state((tau_pos - tau_neg)*0.01)

                        e_old = np.concatenate((e_old, np.zeros((1, self.n_joints))))
                        for k in range(self.n_joints):
                            ccc = self.cereb_control_class
                            # b) Inject robot error in IO
                            # joint position has just been updated to tt with tau(tt - T), so select tt!
                            e_new = ccc.get_error_value(tt + self.T, self.robot.q[k], k)    # self.robot.q[j])
                            e_old[-1, k] = e_new
                            e = e_old[0, k]

                            if self.n_joints == 1: e = e*5.
                            if self.n_joints == 2: e = e*3.
                            if e > 0.:
                                if e > 7.: e = 7.
                                # set the future spike trains (in [tt + T, tt + 2T])
                                self.set_poisson_fr(e, [ccc.Cereb_class.CTX_pops['US_p'][k]], actual_sim_time + self.T)
                            elif e < 0.:
                                if e < -7.: e = -7.
                                # set the future spike trains (in [tt + T, tt + 2T])
                                self.set_poisson_fr(-e, [ccc.Cereb_class.CTX_pops['US_n'][k]], actual_sim_time + self.T)
                        e_old = e_old[1:, :]

                        # update tau value for next step
                        new_tau = calculate_instantaneous_fr(self.nest, self.sd_list_R, self.pop_list_to_robot,
                                                             time=actual_sim_time, T_sample=self.T)
                        # new_tau, t_prev = calculate_instantaneous_burst_activity(self.nest, self.sd_list_R, self.pop_list_to_robot,
                        #                                      time=actual_sim_time, t_prev=t_prev)
                        tau, tau_old = evaluate_fir(tau_old, new_tau.reshape((1, self.u_R_dim)), kernel=kernel_robot)
                        if tau_sol is None:
                            tau_sol = np.array([tau])
                        else:
                            tau_sol = np.concatenate((tau_sol, [tau]), axis=0)

                        # update io
                        new_io = calculate_instantaneous_fr(self.nest, sd_list_io, list_io,
                                                             time=actual_sim_time, T_sample=self.T)
                        io, io_old = evaluate_fir(io_old, new_io.reshape((1, 2*self.n_joints)), kernel=kernel)
                        if io_sol is None:
                            io_sol = np.array([io])
                        else:
                            io_sol = np.concatenate((io_sol, [io]), axis=0)

                    tt = tt + self.T  # update actual time

                # vsl.simple_plot(range(len(q_val)), q_val)
                # vsl.simple_plot(range(len(q_val)), tau_sol[-len(q_val):, :])

                # if settling_time > 0.:
                #     for j, ccc in enumerate(self.cereb_control_class):
                #         pf_pc_syn = self.nest.GetStatus(ccc.Cereb_class.PF_PC_conn)
                #         w = [pf_pc_syn[i]['weight'] for i in range(len(pf_pc_syn))]

        # save the time evolution of mass model state as self.ode_sol variable
        self.ode_sol = ode_sol
        self.u_sol = u_sol
        if self.cereb_control_class is not None:
            self.io_sol = io_sol
            self.tau_sol = tau_sol


    def set_poisson_fr(self, fr, target_pop, time, yT_buf=None):
        """ Set the firing rate for a list of poisson generators (which are pops of neurons) """
        # first, save yT in yT_buf
        if yT_buf is not None:
            yT_buf = np.concatenate((yT_buf, fr), axis=0)
            set_yT = yT_buf[0, :]
            yT_buf = yT_buf[1:, :]    # discard the old elem, which we will set now
        else:
            set_yT = np.tile(fr, [1])

        # bkgroung_fr = np.array([0, 162.5, 162.5, 486., 486., 642.6, 642.6, 700.4])

        for idx, poiss in enumerate(target_pop):
            if set_yT[idx] < 0.:        # solving numerical problem generating negative fr
                # # print(set_yT[idx])
                set_yT[idx] = 0.
            spike_times = generate_poisson_trains(poiss, set_yT[idx], self.T, time, self.rng)  # long as number of neurons in pop
            # spike_times = self.generate_poisson_trains(poiss, set_yT[idx] + bkgroung_fr[idx], self.T, time)  # long as number of neurons in pop
            generator_params = [{"spike_times": s_t, "spike_weights": [1.] * len(s_t)} for s_t in spike_times]
            self.nest.SetStatus(poiss, generator_params)

        return yT_buf

    def attach_io_spike_det(self):
        io_list = []
        sub_pop_IO_len = int(len(self.cereb_control_class.Cereb_class.Cereb_pops['IO']) / self.n_joints)
        for j in range(self.n_joints):
            io_neg = list(self.cereb_control_class.Cereb_class.Cereb_pops['IO'][2 * j * sub_pop_IO_len // 2:(2 * j + 1) * sub_pop_IO_len // 2])
            io_pos = list(self.cereb_control_class.Cereb_class.Cereb_pops['IO'][(2 * j + 1) * sub_pop_IO_len // 2:(2 * j + 2) * sub_pop_IO_len // 2])
            io_list += [io_neg] + [io_pos]

        return utils.attach_spikedetector(self.nest, io_list), io_list


def generate_poisson_trains(poisson, fr, T, time, rand_gen):
    """ Generate a train of spikes as a Poisson process
        In this version we evaluate the number of spikes in
        1 interval, and we place them using linspace        """
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
                    np.linspace(T / (occ * 2), T * (1. - 1. / (occ * 2)), num=occ,
                                endpoint=True)  # place at the centre
                    + rand_gen.normal(0, gaussian_sd(occ), size=occ),  # add gaussian noise
                    1))  # round definition at 0.1 ms
            if occ != 0 else [] for occ in occurrences]  # replace with empty list if no occurrences to be places
        # maybe better with uniform draws?

    if method == 'uniform':
        spike_trains = []
        for occ in occurrences:
            if occ == 0:
                train = []
            elif occ > 0:
                # init_interval_values = time + np.linspace(0, T * (1. - 1. / occ), num=occ, endpoint=True)  #
                init_interval_values = time + np.linspace(0 + 0.06, T - 0.06, num=occ, endpoint=False)  #
                train = init_interval_values + rand_gen.uniform(0, (T - 0.12) / occ, size=occ)
                # print(np.round(train, 1))
            spike_trains += [np.round(train, 1)]

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
                    s_t[d == 0] += 0.1  # move the second element forward
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