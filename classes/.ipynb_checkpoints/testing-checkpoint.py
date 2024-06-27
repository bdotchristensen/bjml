# print out all cards in deck and stop when the shoe is hit
d = Decks()
while len(d) > d.shoe_cutoff:
    c = d.deal()
    print(c)


# deal 4 cards and print the score
h = Hand()

for _ in range(4):
    c = d.deal()
    h.add_card(c)


# deal cards until score of 17 and then print hand
d = Decks()
h = Hand()

while len(d) > d.shoe_cutoff:

    h = Hand()
    
    for _ in range(2):
        c = d.deal()
        h.add_card(c)

    while h.points().min() <= 17:
        c = d.deal()
        h.add_card(c)
    
    print(h.points().min(), '|', h)