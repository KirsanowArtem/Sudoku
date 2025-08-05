from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters

TOKEN = "8029722918:AAE0gjuUZt6pZYGu3ifFYpXNGp-0njJxsWA"


def parse_sudoku(text):
    lines = text.strip().split("\n")

    # Проверка количества строк
    if len(lines) != 9:
        raise ValueError(f"Неверное количество строк: {len(lines)} (должно быть 9)")

    board = []
    for idx, line in enumerate(lines, start=1):
        if len(line.strip()) < 9:
            raise ValueError(f"Неправильная строка номер {idx} ---> {line}")

        row = []
        for ch in line.strip():
            if ch in "123456789":
                row.append(int(ch))
            else:
                row.append(0)
        board.append(row)

    return board


def possible_values(board, r, c):
    if board[r][c] != 0:
        return set()
    nums = set(range(1, 10))
    nums -= set(board[r])
    nums -= {board[i][c] for i in range(9)}
    br, bc = (r // 3) * 3, (c // 3) * 3
    nums -= {board[i][j] for i in range(br, br + 3) for j in range(bc, bc + 3)}
    return nums


def logical_solve(board):
    progress = True
    while progress:
        progress = False
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    opts = possible_values(board, r, c)
                    if len(opts) == 1:
                        board[r][c] = opts.pop()
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


def solve_backtracking(board):
    empty = find_empty(board)
    if not empty:
        return True
    r, c = empty
    for num in range(1, 10):
        if is_valid(board, r, c, num):
            board[r][c] = num
            if solve_backtracking(board):
                return True
            board[r][c] = 0
    return False


def solve_sudoku(board):
    logical_solve(board)
    solve_backtracking(board)
    return board


def board_to_text(board):
    return "\n".join("".join(str(num) if num != 0 else "_" for num in row) for row in board)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        board = parse_sudoku(update.message.text)
        solve_sudoku(board)
        await update.message.reply_text("Решение:\n" + board_to_text(board))
    except ValueError as ve:
        await update.message.reply_text(str(ve))
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")


if __name__ == "__main__":
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Бот запущен!")
    app.run_polling()
