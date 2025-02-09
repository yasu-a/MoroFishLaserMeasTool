from dataclasses import dataclass

from core.tk.color import Color


@dataclass(slots=True)
class ApplicationUIStyle:
    fg_color: Color
    edge_color: Color
    toast_info_bg_color: Color
    toast_info_fg_color: Color
    toast_error_bg_color: Color
    toast_error_fg_color: Color
    message_info_bg_color: Color
    message_info_fg_color: Color
    message_error_bg_color: Color
    message_error_fg_color: Color
    query_fg_color: Color
    query_bg_color: Color
    dialog_active_button_color: Color
    border_normal: Color
    border_abnormal: Color


_DEFAULT_STYLE = ApplicationUIStyle(
    fg_color=Color.BRIGHT_GREEN,
    edge_color=Color.GREEN,
    toast_info_bg_color=Color.GREEN,
    toast_info_fg_color=Color.BRIGHT_WHITE,
    toast_error_bg_color=Color.RED,
    toast_error_fg_color=Color.BRIGHT_WHITE,
    message_info_bg_color=Color.GREEN,
    message_info_fg_color=Color.WHITE,
    message_error_bg_color=Color.RED,
    message_error_fg_color=Color.WHITE,
    query_fg_color=Color.BRIGHT_WHITE,
    query_bg_color=Color.BLUE,
    dialog_active_button_color=Color.BRIGHT_BLACK,
    border_normal=Color.BRIGHT_BLUE,
    border_abnormal=Color.BRIGHT_RED,
)

if __name__ == '__main__':
    import numpy as np
    import cv2

    active_style = _DEFAULT_STYLE

    # preview
    im_lst = []


    def append_preview(fg: Color, bg: Color, text: str):
        image = np.zeros((50, 200, 3), np.uint8)
        cv2.rectangle(image, (10, 10), (190, 40), bg, -1)
        cv2.putText(image, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 1, cv2.LINE_AA)
        im_lst.append(image)


    append_preview(
        fg=active_style.fg_color,
        bg=active_style.edge_color,
        text='Foreground'
    )
    append_preview(
        fg=active_style.toast_info_fg_color,
        bg=active_style.toast_info_bg_color,
        text='Toast info'
    )
    append_preview(
        fg=active_style.toast_error_fg_color,
        bg=active_style.toast_error_bg_color,
        text='Toast error'
    )
    append_preview(
        fg=active_style.message_info_fg_color,
        bg=active_style.message_info_bg_color,
        text='Message info'
    )
    append_preview(
        fg=active_style.message_error_fg_color,
        bg=active_style.message_error_bg_color,
        text='Message error'
    )
    append_preview(
        fg=active_style.query_fg_color,
        bg=active_style.query_bg_color,
        text='Query'
    )

    im = np.vstack(im_lst)
    cv2.imshow('Style Preview', im)
    cv2.waitKey(0)
