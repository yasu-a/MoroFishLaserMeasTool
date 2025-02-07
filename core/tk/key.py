from enum import IntEnum, auto, Enum


class Key(IntEnum):
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()
    G = auto()
    H = auto()
    I = auto()
    J = auto()
    K = auto()
    L = auto()
    M = auto()
    N = auto()
    O = auto()
    P = auto()
    Q = auto()
    R = auto()
    S = auto()
    T = auto()
    U = auto()
    V = auto()
    W = auto()
    X = auto()
    Y = auto()
    Z = auto()
    F1 = auto()
    F2 = auto()
    F3 = auto()
    F4 = auto()
    F5 = auto()
    F6 = auto()
    F7 = auto()
    F8 = auto()
    F9 = auto()
    F10 = auto()
    F11 = auto()
    F12 = auto()
    F13 = auto()
    F14 = auto()
    F15 = auto()
    F16 = auto()
    SPACE = auto()
    TAB = auto()
    ENTER = auto()
    ESCAPE = auto()
    PAGE_UP = auto()
    PAGE_DOWN = auto()
    BACKSPACE = auto()
    DELETE = auto()
    LEFT = auto()
    UP = auto()
    RIGHT = auto()
    DOWN = auto()
    MINUS = auto()
    UNDERSCORE = auto()
    NUM_0 = auto()
    NUM_1 = auto()
    NUM_2 = auto()
    NUM_3 = auto()
    NUM_4 = auto()
    NUM_5 = auto()
    NUM_6 = auto()
    NUM_7 = auto()
    NUM_8 = auto()
    NUM_9 = auto()

    @classmethod
    def printable_char_map(cls) -> "dict[tuple[Key, tuple[Modifier, ...]], str]":
        return {
            (Key.A, ()): "a",
            (Key.B, ()): "b",
            (Key.C, ()): "c",
            (Key.D, ()): "d",
            (Key.E, ()): "e",
            (Key.F, ()): "f",
            (Key.G, ()): "g",
            (Key.H, ()): "h",
            (Key.I, ()): "i",
            (Key.J, ()): "j",
            (Key.K, ()): "k",
            (Key.L, ()): "l",
            (Key.M, ()): "m",
            (Key.N, ()): "n",
            (Key.O, ()): "o",
            (Key.P, ()): "p",
            (Key.Q, ()): "q",
            (Key.R, ()): "r",
            (Key.S, ()): "s",
            (Key.T, ()): "t",
            (Key.U, ()): "u",
            (Key.V, ()): "v",
            (Key.W, ()): "w",
            (Key.X, ()): "x",
            (Key.Y, ()): "y",
            (Key.Z, ()): "z",
            (Key.A, (Modifier.SHIFT,)): "A",
            (Key.B, (Modifier.SHIFT,)): "B",
            (Key.C, (Modifier.SHIFT,)): "C",
            (Key.D, (Modifier.SHIFT,)): "D",
            (Key.E, (Modifier.SHIFT,)): "E",
            (Key.F, (Modifier.SHIFT,)): "F",
            (Key.G, (Modifier.SHIFT,)): "G",
            (Key.H, (Modifier.SHIFT,)): "H",
            (Key.I, (Modifier.SHIFT,)): "I",
            (Key.J, (Modifier.SHIFT,)): "J",
            (Key.K, (Modifier.SHIFT,)): "K",
            (Key.L, (Modifier.SHIFT,)): "L",
            (Key.M, (Modifier.SHIFT,)): "M",
            (Key.N, (Modifier.SHIFT,)): "N",
            (Key.O, (Modifier.SHIFT,)): "O",
            (Key.P, (Modifier.SHIFT,)): "P",
            (Key.Q, (Modifier.SHIFT,)): "Q",
            (Key.R, (Modifier.SHIFT,)): "R",
            (Key.S, (Modifier.SHIFT,)): "S",
            (Key.T, (Modifier.SHIFT,)): "T",
            (Key.U, (Modifier.SHIFT,)): "U",
            (Key.V, (Modifier.SHIFT,)): "V",
            (Key.W, (Modifier.SHIFT,)): "W",
            (Key.X, (Modifier.SHIFT,)): "X",
            (Key.Y, (Modifier.SHIFT,)): "Y",
            (Key.Z, (Modifier.SHIFT,)): "Z",
            (Key.SPACE, ()): " ",
            (Key.MINUS, ()): "-",
            (Key.UNDERSCORE, ()): "_",
            (Key.NUM_0, ()): "0",
            (Key.NUM_1, ()): "1",
            (Key.NUM_2, ()): "2",
            (Key.NUM_3, ()): "3",
            (Key.NUM_4, ()): "4",
            (Key.NUM_5, ()): "5",
            (Key.NUM_6, ()): "6",
            (Key.NUM_7, ()): "7",
            (Key.NUM_8, ()): "8",
            (Key.NUM_9, ()): "9",
        }


class Modifier(Enum):
    SHIFT = 1
    CONTROL = 2


def code_to_key(code: int) -> tuple[Key, tuple[Modifier, ...]] | None:
    if ord("0") <= code <= ord("9"):
        return Key(code - ord("0") + Key.NUM_0), ()

    if ord("a") <= code <= ord("z"):
        return Key(code - ord("a") + Key.A), ()

    if ord("A") <= code <= ord("Z"):
        return Key(code - ord("A") + Key.A), (Modifier.SHIFT,)

    key = {
        ord(" "): Key.SPACE,
        ord("\t"): Key.TAB,
        ord("\r"): Key.ENTER,
        ord("\x1b"): Key.ESCAPE,
        0x210000: Key.PAGE_UP,
        0x220000: Key.PAGE_DOWN,
        0x8: Key.BACKSPACE,
        0x2e0000: Key.DELETE,
        0x250000: Key.LEFT,
        0x260000: Key.UP,
        0x270000: Key.RIGHT,
        0x280000: Key.DOWN,
        ord("-"): Key.MINUS,
        ord("_"): Key.UNDERSCORE,
    }.get(code)
    if key is not None:
        return key, ()

    if 0x1 <= code <= 0x1a:
        return Key(code - 0x1 + Key.A), (Modifier.CONTROL,)

    if (code & 0xf0ffff) == 0x700000:
        return Key((code >> 16) & 0x0f + Key.F1), ()

    return None
