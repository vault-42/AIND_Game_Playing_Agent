# Isolation Game Playing Agent

## Synopsis

This repository contains my submission for the second project of the Udacity Artificial Intelligence Nanodegree program. In this project we game playing agent to play the Isolation game as knight (able to make the same moves as a knight from the game Chess). In order to create the agent I implemented the **Minimax** and **AlphaBeta** search algorithms. These adversarial search algorithms search future player moves to find the best move according to an evaluation heuristic that maximizes the player's positional value and minimizes the opponents positional value. I also evaluated several heuristics and analyzed their performance in the `heuristic_analysis.pdf` file. For this project submission I created a two page summary of the "Mastering the game of Go with deep neural networks and tree search" paper by DeepMind<sup>[1](#myfootnote1)</sup> (please see`research_review.pdf`). The objective of the paper is to summarize the key innovations made by the DeepMind team which allowed their AlphaGo machine to become the best Go player in the world.

## Isolation Game Visualization

![Example game of isolation](viz.gif)

Created by Udacity Staff.

## Code

This project requires Python 3. Project used a pre-packaged Anaconda3 environment provided by the Udacity staff for the artificial intelligence nanodegree program.

* `game_agent.py` - Contains the code for the Isolation playing agent.
* `agent_test.py` - Provided by the Udacity staff for functional testing (run `python agent_test.py`).
* `tournament.py` - Provided by the Udacity staff for performance evaluation of the Isolation playing agent (run `python tournament.py`).

## References
<a name="myfootnote1">[1]</a> Mastering the game of Go with deep neural networks and tree search, by David Silver et als @ https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf
