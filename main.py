
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from environment import KarelEnv
from policy_models import ActorCriticMLP
from algorithms import ActorCritic, SoftActorCritic


from parse_utils import parse_dataset, COMPACT


COMPACT = True

X, y = parse_dataset(levels=["easy", "medium", "hard"], mode='train')
env = KarelEnv(task_space=X)

num_inputs = 16*5 + 8 if COMPACT else 11*16

a2c_policy = ActorCriticMLP(num_inputs, 6, soft_critic=False, hidden_sizes=(256,256))
sac_policy = ActorCriticMLP(num_inputs, 6, soft_critic=True, hidden_sizes=(256,256))
#agent = ActorCritic(policy, env, load_pretrained=True)

a2c_agent = ActorCritic(a2c_policy, env, load_pretrained=False, max_episodes=100000, alpha=0.5, early_stop=7, variant_name='compact') 
sac_agent = SoftActorCritic(sac_policy, env, load_pretrained=False, max_episodes=100000, alpha=0.5, early_stop=7, variant_name='compact')

X_test, y_test = parse_dataset(levels=["easy", "medium", "hard"], mode='val')
val_size = 5000


a2c_agent.train(X, y, data_val=(X_test[:val_size], y_test[:val_size]))
sac_agent.train(X, y, data_val=(X_test[:val_size], y_test[:val_size]))

#agent.train(X, y, data_val=(X_test, y_test))
a2c_agent.evaluate(X_test, y_test, verbose=True)
sac_agent.evaluate(X_test, y_test, verbose=True)


#agent.solve(X_test, y_test)

