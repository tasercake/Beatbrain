import enum


class DataType(enum.Enum):
    AUDIO = 1
    NUMPY = 2
    ARRAY = 2
    IMAGE = 3
    UNKNOWN = 4
    AMBIGUOUS = 5
    ERROR = 6


EXTENSIONS = {
    DataType.AUDIO: [
        "wav",
        "flac",
        "mp3",
        "ogg",
    ],  # TODO: Remove artificial limit on supported audio formats
    DataType.NUMPY: ["npy", "npz"],
    DataType.IMAGE: ["tiff", "exr"],
}
