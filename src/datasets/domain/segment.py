from src.datasets.domain.track import Track


class Segment:
    def __init__(self):
        self.tracks: dict[str, Track] = {}
