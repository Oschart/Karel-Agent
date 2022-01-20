from environment import KarelEnv
from policy_models import ActorCriticMLP
from algorithms import ActorCritic


from parse_utils import parse_dataset

X, y = parse_dataset(levels=["easy", "medium"], mode='train')
env = KarelEnv(task_space=X)

num_inputs = 11*16
policy = ActorCriticMLP(num_inputs, 6, soft_critic=False, hidden_sizes=(256,256))
agent = ActorCritic(policy, env, load_pretrained=True)

#agent.train(X, y)
X_test, y_test = parse_dataset(levels=["medium"], mode='val')

agent.train(X, y, data_val=(X_test, y_test))



#agent.solve(X_test, y_test)

