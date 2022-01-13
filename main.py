from Environment import KarelEnv
from policy_models import ActorCriticMLP
from algorithms import ActorCritic


from parse_utils import parse_dataset
#from PolicyGradientAgent import PolicyGradientAgent

X, y = parse_dataset(data_dir='datasets/data_medium', mode='train')

env = KarelEnv(task_space=X)
X_test, y_test = parse_dataset(data_dir='datasets/data_medium', mode='val')

num_inputs = 11*16
policy = ActorCriticMLP(num_inputs, 6, soft_critic=False, hidden_sizes=(256,256))
agent = ActorCritic(policy, env, max_episodes=24000)
agent.train(X, y)

#agent.train(X, y)

agent.solve(X_test, y_test)

