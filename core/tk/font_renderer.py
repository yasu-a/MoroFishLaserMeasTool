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
        if thickness >= 2:
            mask = cv2.dilate(mask, self._MORPH_MASK, iterations=thickness)

        mask = cv2.resize(mask, None, fx=size / 100, fy=size / 100, interpolation=cv2.INTER_AREA)

        width = int(55 * size / 100)
        height = int(100 * size / 100)

        mask = mask[:height, :width]

        return mask

    def get_mask_shape(self, ch, size: float) -> tuple[int, int]:
        im = self.get_char_mask(ch, size, 1)
        return im.shape[1], im.shape[0]

    def get_char_baseline(self, ch, size: float) -> int:
        return int(100 * size / 100)

    def get_global_baseline(self, size: float) -> int:
        return int(100 * size / 100)


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
        for ch in text:
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

            x += mask.shape[1]
            len_printed += 1

        return text[len_printed:]

    def text_height(self, text: str, size: float) -> int:
        return int(size)

    def text_width_array(self, text: str, size: float) -> np.ndarray:
        return np.array([self._char_factory.get_mask_shape(ch, size)[0] for ch in text]).astype(int)

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

        def flush(self) -> bool:  # False if commited empty line
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
                    self.flush()
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
            return self._planned_lines

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
    ) -> str:  # returns rest of text
        # plan newlines
        lines = self.LineWrapPlanner(
            text,
            size,
            max_width,
            self.text_width_array(text, size),
        ).plan_lines()
        if lines is None:
            return text
        text = "\n".join(lines)

        y_start = pos[1]
        x, y = pos
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
            text = cur_rest + text
            if len(text) == prev_len:
                break
            prev_len = len(text)
            y += height + line_margin
        return text


if __name__ == '__main__':
    import numpy as np
    import cv2


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
            text = """    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec venenatis blandit nulla vitae viverra. Mauris tincidunt cursus massa sed tincidunt. Vivamus lobortis fermentum massa ut vehicula. Nullam id tellus egestas, porta sem eget, ornare nisi. Praesent sapien tellus, elementum eget gravida non, faucibus nec leo. Fusce vitae erat ac sapien sollicitudin pellentesque et at purus. Nunc justo mauris, facilisis eu interdum in, ullamcorper sit amet augue. Curabitur hendrerit, leo eu faucibus efficitur, lacus massa tempus tellus, ac imperdiet felis velit nec ex. Nullam nec sapien pharetra, lacinia arcu placerat, lobortis tellus. Vivamus at lorem vel magna fringilla convallis. Sed egestas semper pretium. Etiam lorem lorem, euismod ut dictum et, efficitur sit amet nisl. Nunc eleifend dignissim ex non lacinia. Etiam vitae tortor elit. Donec egestas consequat elit, ut varius lorem sollicitudin semper. Quisque sagittis neque eget sem consectetur convallis elementum et felis.
    Etiam pulvinar dictum lectus eu sodales. Mauris sodales molestie enim sed faucibus. Etiam ac tellus vel eros dapibus elementum. Mauris sit amet ullamcorper nisi, rhoncus tincidunt sem. Sed sit amet diam consequat, tempor quam vitae, eleifend ante. In sem est, luctus nec nunc vitae, pharetra sagittis urna. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Integer auctor vitae ex sit amet hendrerit. In hac habitasse platea dictumst. Phasellus id ipsum at libero eleifend tincidunt. Cras sed nisi ornare, dignissim neque sit amet, lobortis neque. Ut mattis ut dui ut condimentum.
    Curabitur id eros quis tellus iaculis pretium non id felis. Donec sed venenatis lorem. Mauris et massa nec lorem sollicitudin feugiat. Duis sit amet commodo eros. Nulla ut arcu mauris. Suspendisse semper urna ut enim tincidunt, eget iaculis eros ultricies. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Aliquam a rhoncus lorem, eget vehicula nisi. Aliquam laoreet elit eu pretium feugiat.
    Ut vulputate, nulla vitae auctor feugiat, metus orci suscipit sem, quis facilisis lorem diam non metus. Nam luctus nisi id pretium sagittis. Pellentesque sed ipsum facilisis, molestie velit vel, vulputate eros. Morbi aliquam nisl a rutrum pharetra. In consectetur ex in sollicitudin semper. Vivamus nec libero auctor sem consectetur pellentesque. Quisque nec tincidunt arcu, ac luctus lectus. Nulla convallis turpis purus. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Vestibulum condimentum diam non magna blandit bibendum. Nulla posuere consectetur sem, a ultrices sapien tempus et. Duis ut velit egestas, ullamcorper sem nec, auctor lorem.
    Pellentesque et ligula velit. Curabitur id mauris sed lectus ultrices porttitor. Donec ut sapien ac velit fermentum vestibulum. Etiam rutrum felis ligula, consectetur maximus justo molestie vitae. Sed condimentum dignissim mauris vel egestas. Donec dapibus velit enim, nec aliquam diam efficitur et. Proin mollis in velit ac facilisis. Nulla nec odio consequat, molestie magna in, cursus diam. Mauris imperdiet a nisl sed aliquet."""
            text = """This function uses the same algorithm as the builtin python bisect.bisect_left (side='left') and bisect.bisect_right (side='right') functions, which is also vectorized in the v argument.\nThis function uses the same algorithm as the builtin python bisect.bisect_left (side='left') and bisect.bisect_right (side='right') functions, which is also vectorized in the v argument."""

            max_width = 200
            max_height = 700
            line_ofs = 0

            def mouse_callback(event, x, y, flags, param):
                nonlocal line_ofs
                if event == cv2.EVENT_MOUSEWHEEL:
                    if flags > 0:
                        line_ofs -= 4
                        if line_ofs < 0:
                            line_ofs = 0
                    elif flags < 0:
                        line_ofs += 4

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
                rest = cp.put_text_multiline(
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
                cv2.rectangle(im, (x, y), (x + max_width, y + max_height), (0, 255, 0), 1,
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
