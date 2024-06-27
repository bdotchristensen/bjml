import torch
import torch.nn as nn

from torch.distributions import Binomial

from .cards import *

class ValueBet(nn.Module):
    def __init__(self):
        super().__init__()

        self.act = torch.relu
        
        self.layer1 = nn.Linear(3, 12) # budget, count, decks gone by
        self.layer2 = nn.Linear(12, 12)
        self.layer3 = nn.Linear(12, 1)

    def forward(self, x):
        
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)

        return x
    
class PolicyBet(nn.Module):
    def __init__(self, min_bet = 10, unit_resolution = 5, max_bet = 250):
        super().__init__()
        self.min_bet = min_bet
        self.unit_resolution = unit_resolution
        self.max_bet = max_bet

        self.act = torch.relu
        self.sigmoid = torch.sigmoid
        
        self.layer1 = nn.Linear(3, 12) # budget, count, decks gone by
        self.layer2 = nn.Linear(12, 12)
        self.layer3 = nn.Linear(12, 1)

    def forward(self, state_vector):
        # predicts a poisson lambda
        # budget, count, shoe estimate
        
        x = self.layer1(state_vector)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)
        x = self.sigmoid(x) 

        bindist = Binomial((self.max_bet - self.min_bet) / self.unit_resolution, x)
        sample = bindist.sample()

        return x, sample, self.min_bet + self.unit_resolution * sample

    def possible(self, pot):
        return pot >= self.min_bet
        

class ValueGame(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = torch.relu
        
        self.layer1 = nn.Linear(13 + 13 + 1, 12)
        self.layer2 = nn.Linear(12, 12)
        self.layer3 = nn.Linear(12, 1)     
        
    def forward(self, x):

        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)
        
        return x

class PolicyGame(nn.Module):
    def __init__(self):
        super().__init__()

        self.act = torch.relu
        
        self.layer1 = nn.Linear(13 + 13 + 1, 12)
        self.layer2 = nn.Linear(12, 12)
        self.layer3 = nn.Linear(12, 12)
        self.layer4 = nn.Linear(12, 8) # or however many outputs there are
        
        
    def forward(self, x):

        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)
        x = self.act(x)
        x = self.layer4(x)
        
        return torch.softmax(x, dim = -1)
        
        

class ValueCounter(nn.Module):
    """counter can be both value and action"""
    def __init__(self):
        super().__init__()
        self.act = torch.relu
        
        self.layer1 = nn.Linear(14, 12) # card (one hot) + old current score
        self.layer2 = nn.Linear(12, 12)
        self.layer3 = nn.Linear(12, 1)

    def forward(self, x):

        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)
        return x


class PolicyCounter(nn.Module):
    """counter can be both value and action"""
    def __init__(self):
        super().__init__()
        self.act = torch.relu
        
        self.layer1 = nn.Linear(14, 12) # card (one hot) + old current score
        self.layer2 = nn.Linear(12, 12)
        self.layer3 = nn.Linear(12, 3)

    def forward(self, x):

        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)

        probs = torch.softmax(x, axis = -1)
        return probs, torch.tensor(torch.distributions.Categorical(probs).sample()).reshape(-1, 1) - 1 # -1, 0, or +1
        

class HistoryGame:
    def __init__(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []

    def add_batch(self, list_state_action, reward):
        self.rewards += [reward for _ in list_state_action]

        self.states += [x[0] for x in list_state_action]
        self.probs += [x[1] for x in list_state_action]
        self.actions += [x[2] for x in list_state_action]

    def get_state(self):
        return torch.vstack(self.states)

    def get_reward(self):
        return torch.tensor(self.rewards).reshape(-1, 1)

    def round_k_probs(self):
        return torch.vstack(self.probs)

    def get_actions_index(self):
        return torch.tensor(self.actions).type(torch.int32).ravel()


class HistoryBet:
    def __init__(self):
        self.states = []
        self.probs = []
        self.x = []
        self.rewards = []

    def add(self, state, prob, x, reward):
        
        self.rewards.append(reward)
        self.states.append(state)

        self.probs.append(prob)
        self.x.append(x)

    def get_state(self):
        return torch.vstack(self.states).detach()

    def get_reward(self):
        return torch.tensor(self.rewards).reshape(-1, 1)

    def get_action(self):
        return torch.tensor(self.x)

    def round_k_log_probs(self, pb):
        caps, probs = self.get_binomial_values(pb)
        x = torch.tensor(self.x)
        
        bern = Binomial(caps, probs)
        return bern.log_prob(x)

    def get_binomial_values(self, pb):
        n = torch.tensor([(pb.max_bet - pb.min_bet) / pb.unit_resolution for _ in self.probs])
        p = torch.tensor(self.probs)

        return n, p
        
        


class HistoryCounter:
    def __init__(self):
        self.states = []
        self.probabilities = []
        self.actions = []
        self.rewards = []

    def add_batch(self, list_state_prob_action, r):
        
        self.rewards += [r for _ in list_state_prob_action]

        self.states += [x[0] for x in list_state_prob_action]
        self.probabilities += [x[1] for x in list_state_prob_action]
        self.actions += [x[2] for x in list_state_prob_action]

    def get_state(self):
        return torch.vstack(self.states)

    def get_reward(self):
        return torch.tensor(self.rewards).reshape(-1, 1)

    def get_actions_index(self):
        return torch.vstack(self.actions).type(torch.int32).ravel() + 1

    def round_k_probs(self):
        prob_stack = torch.vstack(self.probabilities)
        actions_stack_index = self.get_actions_index()

        N = prob_stack.shape[0]
        
        return prob_stack[torch.arange(N), actions_stack_index].detach().reshape(-1, 1) # -1 0 1 -> 0 1 2
        




    