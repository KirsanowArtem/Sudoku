import logging
import os
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters
import cv2
import numpy as np

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = "8029722918:AAE0gjuUZt6pZYGu3ifFYpXNGp-0njJxsWA"


def order_points(points):
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def process_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Улучшенная предварительная обработка
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Улучшенный поиск контуров
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Ищем самый большой четырехугольник
        max_area = 0
        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                epsilon = 0.1 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) == 4:
                    max_area = area
                    best_cnt = approx

        if best_cnt is None:
            return None

        # Выравнивание перспективы
        points = np.float32([point[0] for point in best_cnt])
        rect = order_points(points)

        (tl, tr, br, bl) = rect
        width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(gray, M, (width, height))

        # Разделение на клетки с улучшенными параметрами
        cell_size = warped.shape[0] // 9
        cells = []
        for i in range(9):
            row = []
            for j in range(9):
                x = j * cell_size + 2
                y = i * cell_size + 2
                cell = warped[y:y + cell_size - 4, x:x + cell_size - 4]

                # Улучшение контраста клетки
                cell = cv2.bitwise_not(cell)
                cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                row.append(cell)
            cells.append(row)

        return cells
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return None


def recognize_digit(cell):
    # Подсчет ненулевых пикселей
    non_zero = cv2.countNonZero(cell)
    if non_zero < 30:  # Пустая клетка
        return 0

    # Улучшенное распознавание цифр
    cell = cv2.resize(cell, (28, 28))

    # Простые правила для распознавания цифр
    # Центральная область
    center = cell[8:20, 8:20]
    center_pixels = cv2.countNonZero(center)

    # Верхняя область
    top = cell[2:8, 8:20]
    top_pixels = cv2.countNonZero(top)

    # Правая область
    right = cell[8:20, 20:26]
    right_pixels = cv2.countNonZero(right)

    # Логика распознавания цифр
    if center_pixels > 100 and top_pixels > 30 and right_pixels > 30:
        return 8
    elif center_pixels > 100 and top_pixels > 30:
        return 9
    elif center_pixels > 100 and right_pixels > 30:
        return 6
    elif top_pixels > 30 and right_pixels > 30:
        return 3
    elif right_pixels > 30:
        return 1
    elif center_pixels > 100:
        return 0
    elif top_pixels > 30:
        return 7

    return 0  # Если не распознали


def recognize_digits(cells):
    board = [[0 for _ in range(9)] for _ in range(9)]

    for i in range(9):
        for j in range(9):
            board[i][j] = recognize_digit(cells[i][j])

    return board


# Ваши функции решения судоку (оставлены без изменений)
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
    text = ""
    for i in range(9):
        if i % 3 == 0 and i != 0:
            text += "------+-------+------\n"
        for j in range(9):
            if j % 3 == 0 and j != 0:
                text += "| "
            text += f"{board[i][j] if board[i][j] != 0 else '.'} "
        text += "\n"
    return text


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo_file = await update.message.photo[-1].get_file()
        file_path = f"temp_{update.update_id}.jpg"
        await photo_file.download_to_drive(file_path)

        await update.message.reply_text("🔍 Обрабатываю изображение...")
        cells = process_image(file_path)

        if cells is None:
            await update.message.reply_text(
                "❌ Не удалось найти сетку судоку. Пожалуйста, отправьте четкое фото судоку без искажений.")
            os.remove(file_path)
            return

        await update.message.reply_text("🔢 Распознаю цифры...")
        sudoku = recognize_digits(cells)

        await update.message.reply_text("🧩 Решаю судоку...")
        solved_board = [row[:] for row in sudoku]
        solve_sudoku(solved_board)

        original_text = "📷 Исходное судоку:\n" + board_to_text(sudoku)
        solved_text = "\n✅ Решенное судоку:\n" + board_to_text(solved_board)

        await update.message.reply_text(original_text + solved_text)
        os.remove(file_path)

    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("❌ Произошла ошибка при обработке. Пожалуйста, попробуйте другое изображение.")
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = update.message.text
        lines = [line.strip().replace(" ", "") for line in text.split("\n") if line.strip()]

        if len(lines) != 9:
            raise ValueError(f"Неверное количество строк: {len(lines)} (должно быть 9)")

        board = []
        for idx, line in enumerate(lines, start=1):
            if len(line) != 9:
                raise ValueError(f"Неправильная строка номер {idx} ---> {line}")

            row = []
            for ch in line:
                if ch in "123456789":
                    row.append(int(ch))
                elif ch in "0._":
                    row.append(0)
                else:
                    raise ValueError(f"Недопустимый символ '{ch}' в строке {idx}")
            board.append(row)

        solved_board = [row[:] for row in board]
        solve_sudoku(solved_board)

        original_text = "📝 Исходное судоку:\n" + board_to_text(board)
        solved_text = "\n✅ Решенное судоку:\n" + board_to_text(solved_board)

        await update.message.reply_text(original_text + solved_text)
    except ValueError as ve:
        await update.message.reply_text(
            f"❌ {str(ve)}\n\nОтправьте судоку в виде 9 строк по 9 цифр (0 или . для пустых клеток)")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Бот запущен!")
    app.run_polling()