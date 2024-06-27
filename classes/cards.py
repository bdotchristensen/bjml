import torch
import random
from collections import deque

SUITS = ['D', 'H', 'C', 'S']
VALUES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

class Card:
    def __init__(self, num):
        self.suit = num // 13
        self.footprint = num % 13
        self.points = self.get_points()

        self.oho = torch.eye(13)[self.footprint].reshape(1, -1)

    def get_points(self):
        if self.footprint == 0:
            return torch.tensor([1, 11])
        elif self.footprint >= 10:
            return torch.tensor([10])
        else:
            return torch.tensor([self.footprint + 1])

    def __str__(self):
        return VALUES[self.footprint] + SUITS[self.suit] 


class Decks:
    def __init__(self, num = 8, shoe = 0.2):

        self.cards = deque()
        self.shoe_cutoff = num * shoe * 52

        for n in range(num):
            
            for i in range(52):           
                c = Card(i)
                self.cards.append(c)

        self.shuffle()

    def shuffle(self):
        """
        shuffles cards
        """
        random.shuffle(self.cards)

    def deal(self):
        c = self.cards.popleft()
        return c

    def __len__(self):
        return len(self.cards)

    def __str__(self):
        return str(list(map(lambda x: x.__str__(), self.cards)))


class Hand:
    def __init__(self):
        self.cards = [] # empty list right now
        self.can_split = True

    def upcard(self):
        return self.cards[0]

    def add_card(self, c):
        self.cards.append(c)

    def take(self):
        return self.cards.pop(0)

    def points(self):
        total = torch.tensor([0])
        points_list = [c.points for c in self.cards]

        for i, x in enumerate(points_list):
            if i == 0:
                total = x.reshape(-1, 1)

            else:
                total = total + x.reshape(1, -1)
                total = total.reshape(-1, 1)

        return total

    def card_i(self, i): # integrates logic for splitting hands 
        return self.cards[i].get_points()

    def __str__(self):
        return str(list(map(lambda x: x.__str__(), self.cards))) 

    def __len__(self):
        return len(self.cards)