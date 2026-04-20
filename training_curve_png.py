import os
import struct
import zlib


FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "?": ["11110", "00001", "00110", "00100", "00000", "00100", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "11100"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def default_curve_path(log_path):
    absolute_path = os.path.abspath(log_path)
    root, _ = os.path.splitext(absolute_path)
    return f"{root}_curves.png"


def metric_bounds(history, keys):
    values = []
    for item in history:
        for key in keys:
            value = item.get(key)
            if value is not None:
                values.append(float(value))

    if not values:
        return 0.0, 1.0

    lower = min(values)
    upper = max(values)
    if lower == upper:
        padding = max(abs(lower) * 0.1, 0.1)
    else:
        padding = (upper - lower) * 0.1

    return lower - padding, upper + padding


def create_canvas(width, height, color):
    row = bytearray(color * width)
    return [bytearray(row) for _ in range(height)]


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def set_pixel(canvas, x, y, color):
    height = len(canvas)
    width = len(canvas[0]) // 3
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    index = x * 3
    canvas[y][index:index + 3] = bytes(color)


def fill_rect(canvas, x, y, width, height, color):
    height_limit = len(canvas)
    width_limit = len(canvas[0]) // 3
    x_start = clamp(int(x), 0, width_limit)
    y_start = clamp(int(y), 0, height_limit)
    x_end = clamp(int(x + width), 0, width_limit)
    y_end = clamp(int(y + height), 0, height_limit)
    pixel = bytes(color)
    for row_index in range(y_start, y_end):
        row = canvas[row_index]
        for col in range(x_start, x_end):
            offset = col * 3
            row[offset:offset + 3] = pixel


def draw_rect(canvas, x, y, width, height, color, thickness=1):
    fill_rect(canvas, x, y, width, thickness, color)
    fill_rect(canvas, x, y + height - thickness, width, thickness, color)
    fill_rect(canvas, x, y, thickness, height, color)
    fill_rect(canvas, x + width - thickness, y, thickness, height, color)


def draw_circle(canvas, cx, cy, radius, color):
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                set_pixel(canvas, cx + dx, cy + dy, color)


def draw_line(canvas, x0, y0, x1, y1, color, thickness=1):
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy)))
    if steps == 0:
        draw_circle(canvas, int(round(x0)), int(round(y0)), max(1, thickness // 2), color)
        return

    radius = max(1, thickness // 2)
    for step in range(steps + 1):
        ratio = step / steps
        x = int(round(x0 + dx * ratio))
        y = int(round(y0 + dy * ratio))
        draw_circle(canvas, x, y, radius, color)


def draw_text(canvas, x, y, text, color, scale=2, spacing=1):
    cursor_x = int(x)
    upper_text = text.upper()
    for char in upper_text:
        glyph = FONT_5X7.get(char, FONT_5X7["?"])
        for row_index, row_bits in enumerate(glyph):
            for col_index, bit in enumerate(row_bits):
                if bit != "1":
                    continue
                fill_rect(
                    canvas,
                    cursor_x + col_index * scale,
                    int(y) + row_index * scale,
                    scale,
                    scale,
                    color,
                )
        cursor_x += (5 + spacing) * scale


def measure_text(text, scale=2, spacing=1):
    if not text:
        return 0
    return len(text) * (5 + spacing) * scale - spacing * scale


def write_png(canvas, output_path):
    height = len(canvas)
    width = len(canvas[0]) // 3
    raw = b"".join(b"\x00" + bytes(row) for row in canvas)
    compressed = zlib.compress(raw, level=9)

    def chunk(tag, data):
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    png.extend(chunk(b"IDAT", compressed))
    png.extend(chunk(b"IEND", b""))

    with open(output_path, "wb") as output_file:
        output_file.write(png)


def draw_chart(canvas, history, chart_x, chart_y, chart_width, chart_height, metric_title, metric_keys, colors):
    white = (255, 255, 255)
    border = (209, 213, 219)
    grid = (229, 231, 235)
    faint_grid = (243, 244, 246)
    label_color = (107, 114, 128)
    text_color = (17, 24, 39)
    legend_color = (55, 65, 81)

    draw_rect(canvas, chart_x, chart_y, chart_width, chart_height, border, thickness=1)
    fill_rect(canvas, chart_x + 1, chart_y + 1, chart_width - 2, chart_height - 2, white)
    draw_text(canvas, chart_x, chart_y - 28, metric_title, text_color, scale=2)

    max_epoch = max(item["epoch"] for item in history)
    min_y, max_y = metric_bounds(history, metric_keys)
    y_span = max_y - min_y if max_y > min_y else 1.0
    x_denominator = max(1, max_epoch - 1)

    for tick_index in range(5):
        ratio = tick_index / 4
        y_value = max_y - ratio * y_span
        y_pos = int(round(chart_y + ratio * chart_height))
        draw_line(canvas, chart_x, y_pos, chart_x + chart_width, y_pos, grid, thickness=1)
        label = f"{y_value:.3f}"
        draw_text(
            canvas,
            chart_x - measure_text(label, scale=1) - 8,
            y_pos - 4,
            label,
            label_color,
            scale=1,
        )

    epoch_ticks = min(max_epoch, 6)
    for tick_index in range(epoch_ticks):
        epoch = 1 if epoch_ticks == 1 else round(1 + tick_index * (max_epoch - 1) / (epoch_ticks - 1))
        x_pos = int(round(chart_x + ((epoch - 1) / x_denominator) * chart_width))
        draw_line(canvas, x_pos, chart_y, x_pos, chart_y + chart_height, faint_grid, thickness=1)
        label = str(epoch)
        draw_text(
            canvas,
            x_pos - measure_text(label, scale=1) // 2,
            chart_y + chart_height + 8,
            label,
            label_color,
            scale=1,
        )

    epoch_label = "EPOCH"
    draw_text(
        canvas,
        chart_x + chart_width // 2 - measure_text(epoch_label, scale=1) // 2,
        chart_y + chart_height + 28,
        epoch_label,
        legend_color,
        scale=1,
    )

    legend_x = chart_x + chart_width - 160
    legend_y = chart_y - 20
    for series_index, (key, color) in enumerate(zip(metric_keys, colors)):
        y_pos = legend_y + series_index * 18
        draw_line(canvas, legend_x, y_pos, legend_x + 18, y_pos, color, thickness=3)
        draw_text(
            canvas,
            legend_x + 24,
            y_pos - 5,
            key.replace("_", " "),
            legend_color,
            scale=1,
        )

    for key, color in zip(metric_keys, colors):
        points = []
        for item in history:
            value = item.get(key)
            if value is None:
                continue
            x_pos = chart_x + ((item["epoch"] - 1) / x_denominator) * chart_width
            y_pos = chart_y + chart_height - ((value - min_y) / y_span) * chart_height
            points.append((x_pos, y_pos))

        for index in range(len(points) - 1):
            x0, y0 = points[index]
            x1, y1 = points[index + 1]
            draw_line(canvas, x0, y0, x1, y1, color, thickness=3)

        for x_pos, y_pos in points:
            draw_circle(canvas, int(round(x_pos)), int(round(y_pos)), 3, color)


def save_training_curves_png(history, output_path, title):
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    width = 1100
    height = 860
    canvas = create_canvas(width, height, (249, 250, 251))
    title_color = (17, 24, 39)
    subtitle_color = (75, 85, 99)

    best_val_entry = min(
        (item for item in history if item.get("val_loss") is not None),
        key=lambda item: item["val_loss"],
        default=None,
    )
    summary = (
        f"BEST VALIDATION LOSS: {best_val_entry['val_loss']:.4f} AT EPOCH {best_val_entry['epoch']}"
        if best_val_entry is not None
        else "VALIDATION SET NOT PROVIDED SHOWING TRAINING CURVES ONLY"
    )

    draw_text(canvas, 60, 36, title, title_color, scale=3)
    draw_text(canvas, 60, 82, summary, subtitle_color, scale=2)

    draw_chart(
        canvas,
        history,
        chart_x=60,
        chart_y=140,
        chart_width=980,
        chart_height=260,
        metric_title="LOSS CURVE",
        metric_keys=["train_loss", "val_loss"],
        colors=[(37, 99, 235), (220, 38, 38)],
    )
    draw_chart(
        canvas,
        history,
        chart_x=60,
        chart_y=480,
        chart_width=980,
        chart_height=260,
        metric_title="MAE CURVE",
        metric_keys=["train_mae", "val_mae"],
        colors=[(5, 150, 105), (217, 119, 6)],
    )

    write_png(canvas, output_path)
