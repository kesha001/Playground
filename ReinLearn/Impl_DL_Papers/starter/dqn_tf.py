import os
import numpy as np
import tensorflow as tf

class DeepQNetwork(object):
    def __init__(self, lr, n_actions, name, fc1_dims=256,
                input_dims=(210, 160, 4), chkpt_dir='tmp/dqn'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.input_dims = input_dims
        self.chkpt_dir = chkpt_dir

        # instantiate everything into the graph, each network wants to have its own
        self.sess = tf.Session()
        
        # add everything to the graph
        self.build_network()

        # initialise everything in the graph
        self.sess.run(tf.global_variables_initialiser())

        # saving checkpoints, takes long time to train
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, 'deepnet.ckpt')

        # track parameters for each network
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=self.name)
        
    def build_net(self):
        # based on networks name feel placeholder with parameters (inputs, taken aktions, target value for q network)
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims], 
                                        name='inputs')
            # we will take one hot encoding of the actions
            self.actions = tf.placeholder(tf.float32, shape=[None, self.n_action],
                                            name='action_taken')
            # none in shape for batches
            self.q_target = tf.placeholder(tf.float32, shape=[None, self.n_action])

            # layers of network
            conv1 = tf.layers.conv2d(inputs=self.input, filters=32,
                                    kernel_size=(8,8), strides=4, name='conv1',
                                    kernel_initializer=tf.variance_scaling_initializer(scale=2))
            conv1_activated = tf.nn.relu(conv1)                       

            conv2 = tf.layers.conv2d(inputs=conv1_activated, filters=64,
                                kernel_size=(4,4), strides=2, name='conv2',
                                kernel_initializer=tf.variance_scaling_initializer(scale=2))
            conv2_activated = tf.nn.relu(conv2)

            conv3 = tf.layers.conv2d(inputs=conv2_activated, filters=128,
                            kernel_size=(3,3), strides=1, name='conv3',
                            kernel_initializer=tf.variance_scaling_initializer(scale=2))
            conv3_activated = tf.nn.relu(conv3)

            flat = tf.layers.flatten(conv3_activated)
            dense1 = tf.layers.dense(flat, units=self.fc1_dims,
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.variance_scaling_initializer(scale=2))
            # q value - value of a state action pair
            # output of nn
            self.Q_values = tf.layers.dense(dense1, units=self.n_actions,
                            kernel_initializer=tf.variance_scaling_initializer(scale=2))
            # linear activation of network
            # value of q for each action
            self.q = tf.reduce_sum(tf.multiply(self.Q_values, self.actions))

            # loss of squared difference between the q value of outputs ans q target
            self.loss = tf.reduce_mean(tf.square(self.q - self.q_target))

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        # load saved previously saved graph into current session
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        self.saver.save(self.sess, self.checkpoint_file)

class Agent(object):
    # alpha - learning rate, gamma - discounting factor, epsilon - how often it takes to do random action
    # replace_target - how often we want to replace our target network
    def __init__(self, alpha, gamma, mem_size, n_actions, epsilon, batch_size, 
                replace_target=5000, input_dims=(210, 160, 4), 
                q_next_dir='tmp/q_next', q_eval_dir='tmp/q_eval'):
    
        # set of all possible actions
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        # how much we want to discount future rewards
        self.gamma = gamma
        # how many transitions we store in memory
        self.mem_size = mem_size

        self.epsilon = epsilon
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.replace_target = replace_target
        
        # network to tell our agent the value of the next action
        self.q_next = DeepQNetwork(alpha, n_actions, input_dims=input_dims,
                                    name='q_next', chkpt_dir=q_next_dir)

        self.q_eval = DeepQNetwork(alpha, n_actions, input_dims=input_dims,
                                    name='q_eval', chkpt_dir=q_eval_dir)

        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.new_state_memory = np.zeros((self.mem_size, *input_dims))
        # onehot enc. of actions
        self.action_memory = np.zeros((self.mem_size, self.n_actions), dtype=np.int8)

        self.reward_memory = np.zeros(self.mem_size)

        # memory of down flags
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)

    def store_transition(self, state, action, reward, state_, terminal):
        ''' We have fixed size memory and when our actions etc exceed the memory size we will overwrite from the beginning'''

        # counter to keep track of number of memories that are stored
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1


    def choose_action(self, state):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.sess.run(self.q_eval.Q_values,
                                            feed_dict={self.q_eval.input: state})
            action = np.argmax(actions)
        
        return action

    def learn(self):
        if self.mem_cntr % self.replace_target == 0:
            self.update_graph()
        # this allows to randomly sample subset of the memory
        max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size
        # random choice in range max mem of the size batch size
        batch = np.random.choice(max_mem, self.batch_size)
        # state transitions
        state_batch = self.state_memory[batch]
        # actions we took
        action_batch = self.action_memory[batch]
        # convert from one-hot encoding to integers
        action_values = np.array([0,1,2], dtype=np.int8)
        action_indices = np.dot(action_batch, action_values)

        reward_batch = self.reward_memory[batch]
        terminal_batch = self.terminal_memory[batch]
        new_state_batch = self.new_state_memory[batch]

        # calculate values of the current set of states as well as the next set of states
        q_eval = self.q_eval.sess.run(self.q_eval.Q_values,
                                        feed_dict={self.q_eval.input: state_batch})
        # q next takes set of the next transitions
        q_next = self.q_next.sess.run(self.q_next.Q_values,
                                        feed_dict={self.q_next.input: new_state_batch})

        # we want the loss for all non optimal actions to be zero(?)
        q_target = q_eval.copy()
        q_target[:, action_indices] = reward_batch + \
                self.gamma*np.max(q_next, axis=1)*terminal_batch # if the next state is at the end of episode we just want to have the reward else we want to take into account discounted future rewards
        
        _ = self.q_eval.sess.run(self.q_eval.train_op,
                                    feed_dict={self.q_eval.input: state_batch,
                                                self.q_eval.actions: action_batch,
                                                self.q_eval.q_target: q_target}) # q_target which we just calculated
        
        # allow the agent to play some number of moves randomly then decrease eplsilon
        if self.mem_cntr < 100_000:
            if self.epsilon > 0.1:
                self.epsilon *= 0.99999999
            elif self.epsilon <= 0.1:
                self.epsilon = 0.1
    
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    # copy evaluation network to the target network
    def update_graph(self):
        t_params = self.q_next.params
        e_params = self.q_eval.params

        # we need to pass in the session for which we are trying to copy from
        for t, e in zip(t_params, e_params):
            self.q_eval.sess.run(tf.assign(t,e))

    



