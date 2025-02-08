from app_logging import create_logger
from camera_server import CameraServer
from my_app import MyApplication
from record_server import RecordServer
from scene.main_menu import MainScene

_logger = create_logger()
_logger.debug("DEBUG LOG ENABLED")


def main():
    with CameraServer() as camera_server, RecordServer() as record_server:
        app = MyApplication(
            camera_server=camera_server,
            record_server=record_server,
        )
        app.move_to(MainScene())
        app.loop()


if __name__ == '__main__':
    main()
