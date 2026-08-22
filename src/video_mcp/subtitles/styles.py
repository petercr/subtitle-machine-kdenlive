"""Named ASS subtitle style presets."""

from __future__ import annotations

import re
from dataclasses import dataclass

from video_mcp.errors import SubtitleGenerationFailed

_ASS_COLOR = re.compile(r"^&H[0-9A-Fa-f]{8}$")


@dataclass(frozen=True, slots=True)
class ASSStyle:
    """One entry in the ASS ``[V4+ Styles]`` section."""

    name: str = "clean"
    font_name: str = "Arial"
    font_size: int = 48
    primary_color: str = "&H00FFFFFF"
    secondary_color: str = "&H0000FFFF"
    outline_color: str = "&H00000000"
    back_color: str = "&H80000000"
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strike_out: bool = False
    scale_x: int = 100
    scale_y: int = 100
    spacing: int = 0
    angle: int = 0
    border_style: int = 1
    outline: float = 2.0
    shadow: float = 1.0
    alignment: int = 2
    margin_l: int = 40
    margin_r: int = 40
    margin_v: int = 40
    encoding: int = 1

    def __post_init__(self) -> None:
        if not self.name or any(character in self.name for character in ",\r\n"):
            raise SubtitleGenerationFailed("ASS style name must not contain commas or newlines")
        if not self.font_name or any(character in self.font_name for character in ",\r\n"):
            raise SubtitleGenerationFailed("ASS font name must not contain commas or newlines")
        if self.font_size <= 0:
            raise SubtitleGenerationFailed("ASS font size must be greater than zero")
        for color in (
            self.primary_color,
            self.secondary_color,
            self.outline_color,
            self.back_color,
        ):
            if _ASS_COLOR.fullmatch(color) is None:
                raise SubtitleGenerationFailed(f"Invalid ASS color: {color}")
        if self.alignment < 1 or self.alignment > 9:
            raise SubtitleGenerationFailed("ASS alignment must be between 1 and 9")

    def as_ass_line(self) -> str:
        """Return the style in the ASS comma-separated format."""

        return "Style: " + ",".join(
            (
                self.name,
                self.font_name,
                str(self.font_size),
                self.primary_color,
                self.secondary_color,
                self.outline_color,
                self.back_color,
                _ass_bool(self.bold),
                _ass_bool(self.italic),
                _ass_bool(self.underline),
                _ass_bool(self.strike_out),
                str(self.scale_x),
                str(self.scale_y),
                str(self.spacing),
                str(self.angle),
                str(self.border_style),
                _number(self.outline),
                _number(self.shadow),
                str(self.alignment),
                str(self.margin_l),
                str(self.margin_r),
                str(self.margin_v),
                str(self.encoding),
            )
        )


BUILTIN_STYLES: dict[str, ASSStyle] = {"clean": ASSStyle()}


def get_style_preset(name: str) -> ASSStyle:
    """Return a named style preset or raise a user-facing generation error."""

    try:
        return BUILTIN_STYLES[name]
    except KeyError as exc:
        available = ", ".join(sorted(BUILTIN_STYLES))
        raise SubtitleGenerationFailed(
            f"Unknown ASS style '{name}'. Available styles: {available}"
        ) from exc


def _ass_bool(value: bool) -> str:
    return "-1" if value else "0"


def _number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)
