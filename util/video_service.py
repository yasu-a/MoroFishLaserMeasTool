from pathlib import Path

import repo.video
from model.video import Video
from video_reader import VideoReader


def create_video_from_file(src_video_path: Path, video_name: str) -> Video:
    assert src_video_path.suffix == ".mp4", src_video_path
    video = Video(name=video_name)
    repo.video.put(video)
    dst_video_path = repo.video.get_item_path(video_name, "video.mp4")
    dst_video_path.parent.mkdir(parents=True, exist_ok=True)
    src_video_path.rename(dst_video_path)
    return video


def get_video_reader(video: Video) -> VideoReader:
    video_path = repo.video.get_item_path(video.name, "video.mp4")
    return VideoReader(str(video_path))
