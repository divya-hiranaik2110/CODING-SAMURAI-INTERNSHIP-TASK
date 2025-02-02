import math
import random

def print_board(board):
    for row in board:
        print(" ".join(row))
    print()

def check_winner(board):
    for row in board:
        if row.count(row[0]) == 3 and row[0] != ' ':
            return row[0]
    
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != ' ':
            return board[0][col]
    
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != ' ':
        return board[0][0]
    
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != ' ':
        return board[0][2]
    
    return None

def is_moves_left(board):
    for row in board:
        if ' ' in row:
            return True
    return False

def minimax(board, depth, is_maximizing):
    winner = check_winner(board)
    if winner == 'X':
        return -10 + depth
    elif winner == 'O':
        return 10 - depth
    elif not is_moves_left(board):
        return 0
    
    if is_maximizing:
        best_score = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    score = minimax(board, depth + 1, False)
                    board[i][j] = ' '
                    best_score = max(best_score, score)
        return best_score
    else:
        best_score = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    score = minimax(board, depth + 1, True)
                    board[i][j] = ' '
                    best_score = min(best_score, score)
        return best_score

def best_move(board):
    best_score = -math.inf
    move = (-1, -1)
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'O'
                score = minimax(board, 0, False)
                board[i][j] = ' '
                if score > best_score:
                    best_score = score
                    move = (i, j)
    return move

def play_game():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    player_turn = random.choice([True, False])
    predefined_moves = [(0, 0), (1, 1), (0, 1), (1, 0), (0, 2)]
    move_index = 0
    
    while is_moves_left(board) and check_winner(board) is None:
        print_board(board)
        
        if player_turn:
            row, col = predefined_moves[move_index]
            move_index = (move_index + 1) % len(predefined_moves)
            print(f"Player move: {row} {col}")
            if board[row][col] == ' ':
                board[row][col] = 'X'
                player_turn = False
            else:
                print("Invalid move! Skipping.")
        else:
            print("AI is making a move...")
            row, col = best_move(board)
            board[row][col] = 'O'
            player_turn = True
    
    print_board(board)
    winner = check_winner(board)
    if winner:
        print(f"Winner is {winner}!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    play_game()
