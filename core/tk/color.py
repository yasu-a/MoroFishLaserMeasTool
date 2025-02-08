class Color(tuple[int, int, int]):
    def __new__(cls, red: int, green: int, blue: int):
        return super().__new__(cls, (blue, green, red))

    BLACK: "Color"
    RED: "Color"
    GREEN: "Color"
    YELLOW: "Color"
    BLUE: "Color"
    MAGENTA: "Color"
    CYAN: "Color"
    WHITE: "Color"
    BRIGHT_BLACK: "Color"
    GRAY: "Color"
    BRIGHT_RED: "Color"
    BRIGHT_GREEN: "Color"
    BRIGHT_YELLOW: "Color"
    BRIGHT_BLUE: "Color"
    BRIGHT_MAGENTA: "Color"
    BRIGHT_CYAN: "Color"
    BRIGHT_WHITE: "Color"


Color.BLACK = Color(1, 1, 1)
Color.RED = Color(170, 35, 25)
Color.GREEN = Color(10, 80, 0)
Color.YELLOW = Color(255, 199, 6)
Color.BLUE = Color(0, 30, 184)
Color.MAGENTA = Color(118, 38, 113)
Color.CYAN = Color(44, 181, 233)
Color.WHITE = Color(204, 204, 204)
Color.BRIGHT_BLACK = Color(128, 128, 128)
Color.GRAY = Color.BRIGHT_BLACK
Color.BRIGHT_RED = Color(230, 0, 0)
Color.BRIGHT_GREEN = Color(0, 230, 0)
Color.BRIGHT_YELLOW = Color(230, 230, 0)
Color.BRIGHT_BLUE = Color(0, 0, 230)
Color.BRIGHT_MAGENTA = Color(230, 0, 230)
Color.BRIGHT_CYAN = Color(0, 230, 230)
Color.BRIGHT_WHITE = Color(240, 240, 240)

if __name__ == '__main__':
    import cv2
    import numpy as np

    colors = [
        "Color.BLACK",
        "Color.RED",
        "Color.GREEN",
        "Color.YELLOW",
        "Color.BLUE",
        "Color.MAGENTA",
        "Color.CYAN",
        "Color.WHITE",
        "Color.BRIGHT_BLACK",
        "Color.GRAY",
        "Color.BRIGHT_RED",
        "Color.BRIGHT_GREEN",
        "Color.BRIGHT_YELLOW",
        "Color.BRIGHT_BLUE",
        "Color.BRIGHT_MAGENTA",
        "Color.BRIGHT_CYAN",
        "Color.BRIGHT_WHITE",
    ]

    im = np.zeros((400, 400, 3), np.uint8)
    im[:, 200:] = 255
    for i, color in enumerate(colors):
        cv2.putText(
            im,
            f"{color}",
            (10, 20 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            eval(color),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            im,
            f"{color}",
            (210, 20 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            eval(color),
            1,
            cv2.LINE_AA,
        )
    cv2.imshow("Color Preview", im)
    cv2.waitKey(0)
