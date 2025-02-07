from camera_server import CameraServer
from my_app import MyApplication
from record_server import RecordServer


def main():
    with CameraServer() as camera_server, RecordServer() as record_server:
        app = MyApplication(
            camera_server=camera_server,
            record_server=record_server,
        )
        app.loop()


if __name__ == '__main__':
    main()
