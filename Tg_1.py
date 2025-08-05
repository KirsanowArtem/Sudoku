from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder=".")

def possible_values(board, r, c):
    if board[r][c] != 0:
        return set()
    nums = set(range(1, 10))
    nums -= set(board[r])
    nums -= {board[i][c] for i in range(9)}
    br, bc = (r // 3) * 3, (c // 3) * 3
    nums -= {board[i][j] for i in range(br, br+3) for j in range(bc, bc+3)}
    return nums

def get_candidates_state(board):
    return [[sorted(possible_values(board, r, c)) if board[r][c] == 0 else [] for c in range(9)] for r in range(9)]

def logical_solve(board, steps):
    progress = True
    while progress:
        progress = False
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    opts = possible_values(board, r, c)
                    if len(opts) == 1:
                        num = opts.pop()
                        board[r][c] = num
                        steps.append({
                            "type": "logic",
                            "row": r,
                            "col": c,
                            "num": num,
                            "candidates": get_candidates_state(board)
                        })
                        progress = True
    return board

def find_empty(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return r, c
    return None

def is_valid(board, r, c, num):
    if num in board[r]:
        return False
    if num in [board[i][c] for i in range(9)]:
        return False
    br, bc = (r // 3) * 3, (c // 3) * 3
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if board[i][j] == num:
                return False
    return True

def solve_backtracking(board, steps):
    empty = find_empty(board)
    if not empty:
        return True
    r, c = empty
    for num in range(1, 10):
        if is_valid(board, r, c, num):
            board[r][c] = num
            steps.append({
                "type": "guess",
                "row": r,
                "col": c,
                "num": num,
                "candidates": get_candidates_state(board)
            })
            if solve_backtracking(board, steps):
                return True
            board[r][c] = 0
            steps.append({
                "type": "backtrack",
                "row": r,
                "col": c,
                "candidates": get_candidates_state(board)
            })
    return False

@app.route("/")
def index():
    return render_template("sudoku1.html")

@app.route("/solve", methods=["POST"])
def solve():
    data = request.json.get("board", [])
    if not data or len(data) != 9:
        return jsonify({"error": "Неверный формат"}), 400

    board = []
    for row in data:
        if len(row) != 9:
            return jsonify({"error": "Каждая строка должна содержать 9 символов"}), 400
        board.append([int(x) if x.isdigit() else 0 for x in row])

    steps = [{"type": "start", "candidates": get_candidates_state(board)}]
    logical_solve(board, steps)
    solve_backtracking(board, steps)

    return jsonify({"steps": steps})

if __name__ == "__main__":
    app.run(debug=True)
