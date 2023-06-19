from enum import Enum


class Action(Enum):
    WALK = "walk"
    RUN = "run"
    DASH = "dash"
    WALK_BACK = "walk-back"
    WALK_RIGHT = "walk-right"
    WALK_LEFT = "walk-left"
    BOW = "bow"
    BYE = "bye"
    GUIDE = "guide"
    BYEBYE = "byebye"
    RESPOND = "respond"
    CALL = "call"
    KICK = "kick"
    SLASH = "slash"
    DANCE = "dance"

    @classmethod
    def get_all_styles(cls) -> list[str]:
        return [style.value for style in cls]

    def __len__(self) -> int:
        return len(Styles.__members__)


class Styles(Enum):
    ACTIVE = "active"
    NORMAL = "normal"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    PROUD = "proud"
    NOT_CONFIDENT = "not-confident"
    MASCULINITY = "masculinity"
    FEMININE = "feminine"
    CHILDISH = "childish"
    OLD = "old"
    TIRED = "tired"
    MUSICAL = "musical"
    GIANT = "giant"
    CHIMPIRA = "chimpira"

    @classmethod
    def get_all_styles(cls) -> list[str]:
        return [style.value for style in cls]

    def __len__(self) -> int:
        return len(Styles.__members__)
