import cv2

from core.tk.event import KeyEvent, MouseEvent
from core.tk.key import Key, code_to_key, Modifier


class CV2KeyHandler:
    def __init__(self):
        self._active_key_frame_count: dict[Key, int] = {}

    def cv2_wait_key_and_iter_key_events(self, delay):
        code = cv2.waitKeyEx(delay)
        if code >= 0:
            key_result = code_to_key(code)
            if key_result is not None:
                key, modifiers = key_result
                if key in self._active_key_frame_count.keys():
                    self._active_key_frame_count[key] += 1
                else:
                    self._active_key_frame_count[key] = 1
                yield KeyEvent(
                    key=key,
                    modifiers=modifiers,
                    press_count=self._active_key_frame_count[key],
                )
        else:
            for key in self._active_key_frame_count.keys():
                yield KeyEvent(
                    key=key,
                    modifiers=(),
                    press_count=0,
                )
            self._active_key_frame_count.clear()


class CV2MouseHandler:
    def __init__(self):
        self._modifiers = ()
        self._state_l = False
        self._state_r = False
        self._count_l = 0
        self._count_r = 0
        self._move_flag = False
        self._last_xy: tuple[int, int] | None = None

    def _mouse_callback(self, evt, x, y, flags, param) -> None:
        self._last_xy = x, y

        modifiers = []
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            modifiers.append(Modifier.CONTROL)
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            modifiers.append(Modifier.SHIFT)
        self._modifiers = tuple(modifiers)

        if evt == cv2.EVENT_LBUTTONDOWN:
            self._state_l = True
        if evt == cv2.EVENT_LBUTTONUP:
            self._state_l = False
        if evt == cv2.EVENT_RBUTTONDOWN:
            self._state_r = True
        if evt == cv2.EVENT_RBUTTONUP:
            self._state_r = False
        if evt == cv2.EVENT_MOUSEMOVE:
            self._move_flag = True

    def register_callback(self, win_name: str):
        cv2.setMouseCallback(win_name, self._mouse_callback)

    def cv2_iter_mouse_events(self) -> list[MouseEvent]:
        if self._state_l:
            self._count_l += 1
        else:
            self._count_l = 0

        if self._state_r:
            self._count_r += 1
        else:
            self._count_r = 0

        if self._last_xy:
            if self._count_l != 0 or self._count_r != 0:
                yield MouseEvent(
                    x=self._last_xy[0],
                    y=self._last_xy[1],
                    is_move_event=False,
                    right_press_count=self._count_r,
                    left_press_count=self._count_l,
                    modifiers=self._modifiers,
                )
            if self._move_flag:
                self._move_flag = False
                yield MouseEvent(
                    x=self._last_xy[0],
                    y=self._last_xy[1],
                    is_move_event=True,
                    right_press_count=self._count_r,
                    left_press_count=self._count_l,
                    modifiers=self._modifiers,
                )
