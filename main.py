import json

import repo.global_config
from app_logging import create_logger
from camera_server import CameraServer
from my_app import MyApplication
from record_server import RecordServer
from scene.main_menu import MainScene

_logger = create_logger()
_logger.debug("DEBUG LOG ENABLED")


def ask_yesno(title: str, text: str) -> bool:
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno(title, text)
    return result


def main():
    try:
        repo.global_config.get()
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        _logger.error(f"Error reading global configuration: {e}")
        if ask_yesno(
                "Global Config Error",
                "全体設定が破損しています。全体設定を初期化して作り直しますか？",
        ):
            repo.global_config.delete()
        else:
            raise

    with CameraServer() as camera_server, RecordServer() as record_server:
        app = MyApplication(
            camera_server=camera_server,
            record_server=record_server,
        )
        app.move_to(MainScene())
        app.loop()


if __name__ == '__main__':
    main()
