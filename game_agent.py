#import numpy, scipy.spatial
#Note: scipy.spatial not on Udacity's approved libraries list for this project
import numpy

#Helper Functions/Variables
#Applying the opening moves heuristic will cause the program to fail the Udacity open_move test unittest
num_opening_moves = 1
def moveDiff(game,player):
    #totalLegalMovesAI - totalLegalMovesOpp
    return float(len(game.get_legal_moves(player))-len(game.get_legal_moves(game.get_opponent(player))))

def aggressiveMoveDiff(game,player):
    #totalLegalMovesAI - 2*totalLegalMovesOpp
    #Causes AI to "Chase" Opponent
    my_moves=len(game.get_legal_moves(player))
    opp_moves=len(game.get_legal_moves(game.get_opponent(player)))
    return float(my_moves-2*opp_moves)

#Euclidean distance
def dist(x,y): #Inputs: NumpyArrays Returns: double
    #return scipy.spatial.distance.sqeuclidean(x,y)
    return float(numpy.sqrt(numpy.sum((x-y)**2)))
#Manhattan distance
def dist2(x,y): #Inputs: NumpyArrays Returns: double
    #return scipy.spatial.distance.cityblock(x,y)
    return float(sum(numpy.abs(x-y)))
#Chebyshev distance
def dist3(x,y): #Inputs: NumpyArrays Returns: double
    #return scipy.spatial.distance.chebyshev(x,y)
    return float(max(numpy.abs(x[0]-y[0]),numpy.abs(x[1]-y[1])))

def mvToCenter(game,player):
    """
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
        
    Returns
    ---------
    Best Move : Tuple (n,m)
        Move closest to center
    """
    moves = game.get_legal_moves(player)
    best_dist=float("inf")
    best_move=(-1,-1)
    height=game.height
    middle=height//2 #I've already made assumption that board has center
    center=numpy.array((middle,middle))
    for mv in moves:
        dist=dist2(numpy.array(mv),center)
        if dist < best_dist:
            best_dist=dist
            best_move=mv
    return best_move

"""NOTE: Ideas for improvement
* add lookup search to check if I've already calculated this state
* Add more heuristics to analyze:
  ** scores higher when player has more moves and is closer to center than opponent
  ** scores higher when not close to walls or edges (as opposed to just distance from center)
  ** Analyze combination scores (combining different heuristics)
  ** Try switching to an endgame heuristic when there's a limited number of moves left
* Optimize code (loop unrolling, etc). Note that usually at this step instead of doing these optimizations in CPython
  you'd switch to PyPy or a compiled (e.g. C/C++) implementation"""

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    """
    height=game.height
    middle=height//2 #I've already made assumption that board has center
    myPosition = numpy.array(game.get_player_location(player))
    center=numpy.array((middle,middle))
    """
    return moveDiff(game,player)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    #Heuristic: Be close to center
    
    myPosition = numpy.array(game.get_player_location(player))
    center=numpy.array((game.height//2,game.width//2))

    return -dist2(myPosition,center)+aggressiveMoveDiff(game,player)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    #myPosition = numpy.array(game.get_player_location(player))
    #center=numpy.array((game.height//2,game.width//2))
    return aggressiveMoveDiff(game,player)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        board_moves = game.height*game.width-len(game.get_blank_spaces())

        if board_moves <= num_opening_moves: #Opening Moves
            #I'm applying a heuristic that assumes a board that has a center (oddnum x oddnum) board size
            #This assumption is in the lectures.
            """Opening Heuristic:
            Moves as close to center as possible
            """
            #Opening moves heuristic improvement idea: make a table of best possible opening moves to select from
            return mvToCenter(game,self)
        else: #Not opening moves
            moves = game.get_legal_moves()
            best_score = float('-inf')
            if not moves:
                return (-1,-1)
            best_move = moves[0]
            for mv in moves:
                next_level = game.forecast_move(mv)
                score = self.minVal(next_level, depth-1)
                if score > best_score:
                    best_move=mv
                    best_score=score
            return best_move

    def minVal(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        #if out of moves or greater than max depth return score
        moves = game.get_legal_moves()
        if not moves or depth <= 0:
            return self.score(game, self)
        best_score = float('inf')
        for mv in moves:
            next_level = game.forecast_move(mv)
            score = self.maxVal(next_level, depth-1)
            if score < best_score:
                best_score = score
        return best_score
        
    def maxVal(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        #If out of moves or greater than max depth return score
        moves = game.get_legal_moves()
        if not moves or depth <= 0:
            return self.score(game, self)
        best_score = float('-inf')
        for mv in moves:
            next_level = game.forecast_move(mv)
            score = self.minVal(next_level, depth-1)
            if score > best_score:
                best_score = score
        return best_score
                
class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            depth=1
            while True:
                best_move=self.alphabeta(game, depth)
                depth=depth+1

        except SearchTimeout:
            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        board_moves = game.height*game.width-len(game.get_blank_spaces())

        if board_moves <= num_opening_moves: #Opening Moves
            #I'm applying a heuristic that assumes a board that has a center (oddnum x oddnum) board size
            #This assumption is in the lectures.
            """Opening Heuristic:
            Moves as close to center as possible
            """
            #Opening moves heuristic improvement idea: make a table of best possible opening moves to select from
            return mvToCenter(game,self)
        #else it is not opening moves
        moves = game.get_legal_moves()
        if not moves:
            return (-1,-1)
        best_score = float('-inf')
        best_move = moves[0]
        for mv in moves:
            next_level = game.forecast_move(mv)
            score = self.minVal(next_level,alpha,beta,depth-1)
            if score > best_score:
                best_move=mv
                best_score=score
            alpha=max(alpha,best_score)
            if beta <= alpha:
                break
        return best_move

    def minVal(self, game, alpha, beta, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        #if out of moves or greater than max depth return score
        moves = game.get_legal_moves()
        if not moves or depth <= 0:
            return self.score(game,self)
        best_score = float('inf')
        for mv in moves:
            next_level = game.forecast_move(mv)
            best_score = min(best_score, self.maxVal(next_level,alpha,beta,depth-1))
            beta=min(beta,best_score)
            if beta <= alpha:
                break
        return best_score
        
    def maxVal(self, game, alpha, beta, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        #If out of moves or greater than max depth return score
        moves = game.get_legal_moves()
        if moves == [] or depth <= 0:
            return self.score(game,self)
        best_score = float('-inf')
        for mv in moves:
            next_level = game.forecast_move(mv)
            best_score = max(best_score, self.minVal(next_level,alpha,beta,depth-1))
            alpha=max(alpha,best_score)
            if beta <= alpha:
                break
        return best_score