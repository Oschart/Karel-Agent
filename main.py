from Environment import Environment
from parse_utils import parse_dataset
from PolicyGradientAgent import PolicyGradientAgent

env = Environment()
X, y = parse_dataset(data_dir='datasets/data', mode='train')

X_test, y_test = parse_dataset(data_dir='datasets/data_medium', mode='val')

agent = PolicyGradientAgent(pretrained=True, level='hard')

#agent.train(X, y)

agent.solve(X_test, y_test)
