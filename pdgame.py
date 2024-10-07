# %%
import axelrod as axl
# import matplotlib as plt
# from numba.core.cpu_options import InlineOptions
# import numpy as np

# %%
players = (axl.Cooperator(), axl.Random())
match = axl.Match(players, turns=5)
results = match.play()
print(results)
# %%
players = (axl.Cooperator(), axl.Random())
match = axl.Match(players, turns=5)
results = match.play()
print(results)

# %%
players = (axl.Cooperator(), axl.Random())
match = axl.Match(players, turns=5)
results = match.play()

# %%
scores = match.scores()
print(scores)
# %%
axl.game.Game()

# %%
match.final_score()
# %%
match.final_score_per_turn()

# %%
match.winner()
# EXERCISE
# Use the Match class to create the following matches:

# 5 turns match Cooperator vs Defector
# 10 turns match Tit For Tat vs Grumpy
# Creating Tournaments
# Remember the library was created to study the interactions between strategies in a round robin tournament. A tournament where each strategy plays against all opponents. The strategy with the highest score is the winner of the tournament. Here we will cover how we can easily create a very simple IPD tournament.

# Here we create a list of players.

# Reminder: The full list of all the implemented strategies can be found here.
# %%
players = [axl.Cooperator(), axl.Defector(), axl.Random(),
           axl.TitForTat(), axl.Grumpy(), axl.Alternator()]

# %%
tournament = axl.Tournament(players=players)
tournament.turns # default value of turns

# %%
tournament.repetitions # default value of repetitions

# %%
results = tournament.play()

# %%
winners = results.ranked_names
print(winners)

# %%
scores = results.scores
print(scores)

# %%
for i, player in enumerate(players):
    print(f'{player.name}:', scores[i])
    print("========================================================================")


# %%
plot = axl.Plot(results)
p = plot.boxplot()
p.show()
