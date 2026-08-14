"""Welcome panel."""

from __future__ import annotations

import random

from rich.console import Console
from rich.panel import Panel

WelcomeQuote = tuple[str, str]

WELCOME_QUOTES: list[WelcomeQuote] = [
    ("The spice must flow.", "香料必须流通。"),
    (
        "The spice extends life. The spice expands consciousness. The spice is vital to space travel.",
        "香料延续生命。香料扩展意识。香料是星际旅行的关键。",
    ),
    ("He who controls the spice controls the universe.", "掌控香料者，即掌控宇宙。"),
    ("The spice... I can see it.", "香料……我能看见它了。"),
    ("Without the spice, there is no commerce in the universe.", "没有香料，宇宙就没有贸易。"),
    ("Power over spice is power over all.", "掌控香料即掌控一切。"),
    ("Deep truths arrive quietly.", "深处的真相总是静静抵达。"),
    ("A beginning is the time for taking the most delicate care.", "万事开头时，最需精心呵护。"),
    ("Survival is the ability to swim in strange water.", "生存，就是在陌生水域中游泳的能力。"),
    ("The people who can destroy a thing, they control it.", "能毁灭某物的人，才是真正控制它的人。"),
    ("Without change, something sleeps inside us, and seldom awakens.", "没有变化，内心某些东西便沉睡不醒。"),
    ("We must join it. We must flow with it.", "我们必须顺势而行。"),
    ("This is only the beginning.", "这仅仅是个开始。"),
]

def print_welcome(console: Console) -> None:
    english, chinese = random.choice(WELCOME_QUOTES)
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]{english}[/bold]",
                    f"[cyan]{chinese}[/cyan]",
                    "",
                    "[cyan]Type /exit or /quit to leave.[/cyan]",
                ]
            ),
            title="[bold magenta]Spice[/bold magenta]",
            border_style="magenta",
            padding=(1, 4),
        )
    )

def print_compact_welcome(console: Console) -> None:
    english, chinese = random.choice(WELCOME_QUOTES)
    console.print(
        Panel.fit(
            "\n".join([f"[bold]{english}[/bold]", f"[cyan]{chinese}[/cyan]"]),
            border_style="magenta",
            padding=(0, 2),
        )
    )
