import itertools
import string
import time
from abc import ABC, abstractmethod
from collections import deque
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from app_logging import create_logger
from core.tk.color import Color


class AbstractCharFactory(ABC):
    @abstractmethod
    def get_char_mask(self, ch, size: float, thickness: int):
        raise NotImplementedError()

    @abstractmethod
    def get_mask_shape(self, ch, size: float) -> tuple[int, int]:
        raise NotImplementedError()

    @abstractmethod
    def get_char_baseline(self, ch, size: float) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_global_baseline(self, size: float) -> int:
        raise NotImplementedError()

    @abstractmethod
    def char_gap_correction(self, size: float) -> int:
        raise NotImplementedError()


class ConsolaCharFactory(AbstractCharFactory):
    _FONT_NAME = "consola.ttf"

    def __init__(self):
        self._mask_cache: dict[str, np.ndarray] = {}

    def _create_char_mask(self, ch) -> np.ndarray:
        mask = np.zeros((200, 200), np.uint8)
        mask_pil = Image.fromarray(mask)
        draw = ImageDraw.Draw(mask_pil)
        draw.text(
            (0, 0),
            ch,
            255,
            font=ImageFont.truetype(self._FONT_NAME, 100),
        )
        mask = np.array(mask_pil)
        return mask

    _MORPH_MASK = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    @lru_cache(maxsize=65536)
    def get_char_mask(self, ch, size: float, thickness: int):
        if ch not in self._mask_cache:
            self._mask_cache[ch] = self._create_char_mask(ch)

        mask = self._mask_cache[ch]
        mask = np.pad(mask, ((5, 5), (5, 5)))
        if thickness >= 2:
            mask = cv2.dilate(mask, self._MORPH_MASK, iterations=thickness - 1)

        mask = cv2.resize(mask, None, fx=size / 100, fy=size / 100, interpolation=cv2.INTER_AREA)

        height = int(100 * size / 100)
        width = int(70 * size / 100)
        mask = mask[:height, :width]
        mask.setflags(write=False)
        return mask

    def get_mask_shape(self, ch, size: float) -> tuple[int, int]:
        im = self.get_char_mask(ch, size, 1)
        return im.shape[1], im.shape[0]

    def get_char_baseline(self, ch, size: float) -> int:
        return int(100 * size / 100)

    def get_global_baseline(self, size: float) -> int:
        return int(100 * size / 100)

    def char_gap_correction(self, size: float) -> int:
        return int(-2 * size / 15)


class CharPrinter:
    _logger = create_logger()

    def __init__(self, char_factory: AbstractCharFactory):
        self._char_factory = char_factory

    def put_text(
            self,
            im: np.ndarray,
            pos: tuple[int, int],
            text: str,
            size: float,
            color: Color,
            *,
            thickness: int = 1,
            max_width: int = None,
    ) -> str:  # returns rest of text
        # print
        x_start = pos[0]
        x, y = pos
        global_baseline = self._char_factory.get_global_baseline(size)
        warned = False
        len_printed = 0
        width_array = self.text_width_array(text, size)
        for i, ch in enumerate(text):
            mask = self._char_factory.get_char_mask(ch, size, thickness)
            char_baseline = self._char_factory.get_char_baseline(ch, size)
            x_ofs = x
            y_ofs = y + global_baseline - char_baseline
            x_end = x_ofs + mask.shape[1]
            y_end = y_ofs + mask.shape[0]
            if max_width is not None:
                if x_start + max_width < x_end:
                    break

            s = slice(y_ofs, y_end), slice(x_ofs, x_end)
            if mask.shape[:2] != im[s].shape[:2]:
                if not warned:
                    self._logger.warning(f"Character outside of image ignored\ntext={text}")
                    warned = True
                continue

            # blending
            reg = im[s].astype(np.uint16)
            reg = (255 - mask)[:, :, None] * reg + mask[:, :, None] * color
            reg >>= 8
            im[s] = reg.astype(np.uint8)

            x += width_array[i]
            len_printed += 1

        return text[len_printed:]

    # noinspection PyUnusedLocal
    def text_height(self, text: str, size: float) -> int:
        return int(size)

    def text_width_array(self, text: str, size: float) -> np.ndarray:
        a = np.array([
            self._char_factory.get_mask_shape(ch, size)[0]
            + self._char_factory.char_gap_correction(size)
            for ch in text
        ]).astype(int)
        a.setflags(write=False)
        return a

    class LineWrapPlanner:
        def __init__(self, text: str, size: float, max_width: int, char_width_array: np.ndarray):
            self._text = text
            self._size = size
            self._max_width = max_width
            self._char_width_array = char_width_array

            self._pointer = 0
            self._cur_chars = deque([])
            self._cur_w = 0

            self._planned_lines: list[str] = []

        def has_char(self) -> bool:
            return self._pointer < len(self._text)

        def read_char(self) -> tuple[str, int]:  # char and width
            ch = self._text[self._pointer]
            self._cur_chars.append(ch)
            width = int(self._char_width_array[self._pointer])
            self._cur_w += width
            self._pointer += 1
            return ch, width

        def unread_char(self, n: int) -> None:
            while n >= 1:
                self._pointer -= 1
                self._cur_chars.pop()
                width = self._char_width_array[self._pointer]
                self._cur_w -= width
                n -= 1

        def find_backward(self, target_ch, max_unread) -> int | None:  # number of unread
            for i in range(1, max_unread + 1):
                j = self._pointer - i
                if self._pointer - j < 0:
                    break
                if self._text[j] == target_ch:
                    return i
            return None

        def is_over_line_width(self) -> bool:
            return self._cur_w > self._max_width

        def flush(self, ignore_empty=False) -> bool:  # False if commited empty line
            if not self._cur_chars and ignore_empty:
                return bool(self._cur_chars)
            self._planned_lines.append("".join(self._cur_chars))
            self._cur_chars.clear()
            self._cur_w = 0
            return bool(self._planned_lines[-1])

        def discard(self):
            self._cur_chars.clear()
            self._cur_w = 0

        def plan_lines(self) -> list[str] | None:  # None if failure
            max_unread = min(int(self._max_width // 2 + 1), 10)

            while self.has_char():
                ch, _ = self.read_char()
                if ch == "\n":
                    self.unread_char(1)
                    self.flush()
                    self.read_char()
                    self.discard()
                if self.is_over_line_width():
                    n_unread = self.find_backward(" ", max_unread=max_unread)
                    if n_unread is not None:
                        self.unread_char(n_unread)
                        if not self.flush():
                            return None
                        self.read_char()  # space
                        self.discard()
                    else:
                        self.unread_char(1)
                        if not self.flush():
                            return None
            self.flush(ignore_empty=True)
            return self._planned_lines

    @lru_cache(maxsize=65536)
    def plan_line_wrap(self, text: str, size: float, max_width: int):
        lines = self.LineWrapPlanner(
            text,
            size,
            max_width,
            self.text_width_array(text, size),
        ).plan_lines()
        if lines is None:
            self._logger.warning(f"Failed to plan line wrap\ntext={text}")
            return text, 0
        text = "\n".join(lines)
        return text

    def put_text_multiline(
            self,
            im: np.ndarray,
            pos: tuple[int, int],
            text: str,
            size: float,
            color: Color,
            *,
            thickness: int = 1,
            max_width: int = None,
            max_height: int = None,
            line_margin: int = 1,
    ) -> tuple[str, int]:  # returns rest of text and height
        # plan newlines
        text = self.plan_line_wrap(text, size, max_width)

        y_start = pos[1]
        x, y = pos
        y_bottom = y_start
        prev_len = len(text)
        while text:
            if text[0] == "\n":
                text = text[1:]
            next_newline = text.find("\n")
            text_cur = text[:next_newline] if next_newline >= 0 else text
            text = text[next_newline:] if next_newline >= 0 else ""
            height = self.text_height(text_cur, size)
            if max_height is not None and y + height > y_start + max_height:
                break
            cur_rest = self.put_text(
                im,
                (x, y),
                text_cur,
                size,
                color,
                thickness=thickness,
                max_width=max_width,
            )
            y_bottom = y + height
            text = cur_rest + text
            if len(text) == prev_len:
                break
            prev_len = len(text)
            y += height + line_margin
        return text, y_bottom - y_start

    def get_cache_stat(self):
        dct = {}
        for name in itertools.chain(dir(self), dir(self._char_factory)):
            obj = getattr(self, name) if name in dir(self) else getattr(self._char_factory, name)
            if not hasattr(obj, "cache_info"):
                continue
            cache_info_obj = getattr(obj, "cache_info")
            if not callable(cache_info_obj):
                continue
            cache_info = cache_info_obj()
            dct[name] = cache_info
        return dct


if __name__ == '__main__':
    import numpy as np
    import cv2


    # noinspection PyUnusedLocal
    def main():
        average = None

        cf = ConsolaCharFactory()
        cp = CharPrinter(cf)
        size = 15
        x, y = 10, 50

        testcase = "multiline"
        if testcase == "single":
            text = string.printable
            for i in itertools.count():
                im = np.zeros((200, 1600, 3), np.uint8)

                ts = time.perf_counter()

                w, h = cp.text_width_array(text, size).sum(), cp.text_height(text, size)
                cp.put_text(im, (x, y), text, size, Color.BRIGHT_RED)
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

                te = time.perf_counter()
                t = te - ts
                if i > 10:
                    if average is None:
                        average = t
                    else:
                        average = average * 0.95 + t * 0.05

                if i % 20 == 0:
                    if average is not None:
                        print(f"{1 / average:.2f} FPS")

                cv2.imshow("win", im)
                if cv2.waitKey(1) == ord("q"):
                    break
        elif testcase == "multiline":
            text = """A\nBB\nCCC\nDDDDDDDDDDDD\n\nThis function uses the same algorithm as the builtin python bisect.bisect_left (side='left') and bisect.bisect_right (side='right') functions, which is also vectorized in the v argument.\nThis function uses the same algorithm as the builtin python bisect.bisect_left (side='left') and bisect.bisect_right (side='right') functions, which is also vectorized in the v argument."""

            max_width = 200
            max_height = 700
            line_ofs = 0

            # noinspection PyShadowingNames
            def mouse_callback(event, x, y, flags, param):
                nonlocal line_ofs
                if event == cv2.EVENT_MOUSEWHEEL:
                    if flags > 0:
                        line_ofs -= 1
                        if line_ofs < 0:
                            line_ofs = 0
                    elif flags < 0:
                        line_ofs += 1

            cv2.namedWindow("win")
            cv2.setMouseCallback("win", mouse_callback)

            for i in itertools.count():
                im = np.zeros((800, 1600, 3), np.uint8)
                im[...] = 255

                ts = time.perf_counter()

                text_now = "\n".join(text.split("\n")[line_ofs:])
                # cp.put_text_multiline(
                #     im,
                #     (x, y),
                #     text_now,
                #     size,
                #     Color.BRIGHT_WHITE,
                #     thickness=2,
                #     max_width=max_width,
                #     max_height=max_height,
                # )
                rest, h = cp.put_text_multiline(
                    im,
                    (x, y),
                    text_now,
                    size,
                    Color.BRIGHT_RED,
                    thickness=1,
                    max_width=max_width,
                    max_height=max_height,
                )
                if i == 0:
                    print(repr(rest))
                cv2.rectangle(im, (x, y), (x + max_width, y + h), (0, 255, 0), 1,
                              cv2.LINE_AA)

                te = time.perf_counter()
                t = te - ts
                if i > 10:
                    if average is None:
                        average = t
                    else:
                        average = average * 0.95 + t * 0.05

                if i % 20 == 0:
                    if average is not None:
                        print(f"{1 / average:.2f} FPS")

                cv2.imshow("win", im)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                elif key == ord("a"):
                    max_width = max(1, max_width - 1)
                elif key == ord("A"):
                    max_width = max(1, max_width - 10)
                elif key == ord("d"):
                    max_width += 1
                elif key == ord("D"):
                    max_width += 10


    main()
