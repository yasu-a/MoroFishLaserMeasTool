from dataclasses import dataclass

from app_tk.key import Key, Modifier


@dataclass(frozen=True)
class KeyEvent:
    key: Key
    modifiers: tuple[Modifier, ...]
    press_count: int  # 0: key-up, positive>0: count key press frames

    @property
    def down(self) -> bool:
        return self.press_count == 1

    @property
    def up(self) -> bool:
        return self.press_count == 0

    @property
    def pressed(self) -> bool:
        return self.press_count > 0

    @property
    def enter(self) -> bool:
        return self.press_count >= 9 and self.press_count % 3 == 0


@dataclass(frozen=True)
class MouseEvent:
    x: int
    y: int
    is_move_event: bool
    right_press_count: int
    left_press_count: int
    modifiers: tuple[Modifier, ...]

    @property
    def left_drag(self) -> bool:
        if not self.is_move_event:
            return False
        return self.left_press_count != 0

    @property
    def right_drag(self) -> bool:
        if not self.is_move_event:
            return False
        return self.right_press_count == 1

    @property
    def move(self) -> bool:
        if not self.is_move_event:
            return False
        return not self.left_drag and not self.right_drag

    @property
    def right_down(self) -> bool:
        if self.is_move_event:
            return False
        return self.right_press_count == 1

    @property
    def right_up(self) -> bool:
        if self.is_move_event:
            return False
        return self.right_press_count == 0

    @property
    def left_down(self) -> bool:
        if self.is_move_event:
            return False
        return self.left_press_count == 1
