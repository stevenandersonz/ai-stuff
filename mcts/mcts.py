import random
import math
C = 1.4

def ucb_score(node, parent_visits, exploration_constant):
    if node.visits == 0:
        return float('inf')
    exploitation_term = node.wins / node.visits
    exploration_term = math.sqrt(math.log(parent_visits) / node.visits)
    return exploitation_term + exploration_constant * exploration_term

class Node: 
    def __init__(self, state, parent=None, position=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.position = position
        self.visits = 0
        self.wins = 0

    def select(self):
        ucb_scores = [ucb_score(child, self.visits, C) for child in self.children]
        max_ucb = max(ucb_scores)
        max_indices = [i for i in range(len(ucb_scores)) if ucb_scores[i] == max_ucb]
        return self.children[random.choice(max_indices)]

    def expand(self):
        for move in self.state.get_possible_moves():
            copy = self.state.copy()
            copy.make_move(move)
            child = Node(copy, self, move)
            self.children.append(child)

    def update(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.update(result)
        

class TicTacToe:
    def __init__(self):
        self.board = [['-' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'

    def copy(self):
        copy_game = TicTacToe()
        copy_game.board = [row[:] for row in self.board]
        copy_game.current_player = self.current_player
        return copy_game

    def get_possible_moves(self):
        moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == '-':
                    moves.append((row, col))
        return moves

    def make_move(self, move):
        row, col = move
        self.board[row][col] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def is_win(self, player):
        for row in range(3):
            if all(self.board[row][col] == player for col in range(3)):
                return True
        for col in range(3):
            if all(self.board[row][col] == player for row in range(3)):
                return True
        if all(self.board[i][i] == player for i in range(3)):
            return True
        if all(self.board[i][2-i] == player for i in range(3)):
            return True
        return False

    def is_draw(self):
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == '-':
                    return False
        return True

    def is_terminal(self):
        return self.is_win('X') or self.is_win('O') or self.is_draw()

    def get_winner(self):
        if self.is_win('X'):
            return 'X'
        elif self.is_win('O'):
            return 'O'
        else:
            return None

    def print_board(self):
        for row in self.board:
            print(' '.join(row))

def simulate_game(game, debug=False):
    while not game.is_terminal():
        game.make_move(random.choice(game.get_possible_moves()))
    winner = game.get_winner()
    if winner == 'X':
        return -1
    elif winner == 'O':
        return 1
    return 0

def print_tree(node, depth=0):
    if node is None:
        return
    indent = ' ' * depth
    print(f"{indent} {node.position}")
    print(f"{indent} v:{node.visits} w:{node.wins}")
    for child in node.children:
        print_tree(child, depth + 2)

def monte_carlo_tree_search(game, num_iterations, debug=False):
    root = Node(game)
    for _ in range(num_iterations):
        node = root
        while node.children:
            node = node.select()
        if not node.visits:
            node.expand()
        result = simulate_game(node.state.copy(), debug=debug)
        node.update(result)
    if debug:
        print_tree(root)
    bestNode = max(root.children, key=lambda node: node.visits)
    return bestNode.position

game = TicTacToe()



while not game.is_terminal():
    if game.current_player == 'X':
        row = int(input("Enter row: "))
        col = int(input("Enter column: "))
        game.make_move((row, col))
    else:
        move = monte_carlo_tree_search(game, 10000, False)
        game.make_move(move)
    game.print_board()

winner = game.get_winner()
if winner:
    print(f"{winner} wins!")
else:
    print("It's a draw!")