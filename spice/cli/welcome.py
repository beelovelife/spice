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
    ("The spice was life on Arrakis.", "香料就是厄拉科斯上的生命。"),
    ("Power over spice is power over all.", "掌控香料即掌控一切。"),
    (
        "I must not fear. Fear is the mind-killer. Fear is the little-death that brings total obliteration. "
        "I will face my fear. I will permit it to pass over me and through me. And when it has gone past, "
        "I will turn the inner eye to see its path. Where the fear has gone there will be nothing. "
        "Only I will remain.",
        "我绝不能恐惧。恐惧是思维的杀手。恐惧是带来彻底毁灭的小小死亡。我将面对恐惧。我将允许它越过我、穿过我。当它过去之后，"
        "我将转动内心之眼审视它的轨迹。恐惧所过之处，空无一物。唯我独存。",
    ),
    ("Fear is the mind-killer.", "恐惧是思维的杀手。"),
    (
        "The mystery of life isn't a problem to solve, but a reality to experience.",
        "生命的奥秘不是一个待解的问题，而是一种需要经历的现实。",
    ),
    ("Deep truths arrive quietly.", "深处的真相总是静静抵达。"),
    (
        "The future is not fixed. The best prophet is the one who sees the fewest possible futures.",
        "未来并非注定。最好的预言家是能看到最少可能性的人。",
    ),
    ("There is no escape - we pay for the violence of our ancestors.", "无处可逃--我们在为祖先的暴行偿还代价。"),
    ("A beginning is the time for taking the most delicate care.", "万事开头时，最需精心呵护。"),
    ("What do you despise? By this are you truly known.", "你鄙视什么？这才真正定义了你。"),
    ("God created Arrakis to train the faithful.", "上帝创造厄拉科斯，是为了磨炼信徒。"),
    ("The desert takes the weak.", "沙漠吞噬弱者。"),
    ("Polish comes from the cities; wisdom from the desert.", "优雅来自城市，智慧来自沙漠。"),
    ("Survival is the ability to swim in strange water.", "生存，就是在陌生水域中游泳的能力。"),
    ("My road leads into the desert.", "我的路通向沙漠。"),
    ("A great man doesn't seek to lead. He is called to it.", "伟人不寻求领导地位，而是被召唤至此。"),
    ("Plans within plans within plans.", "计中计中计。"),
    ("The slow blade penetrates the shield.", "慢刀穿透护盾。"),
    ("Power attracts the corruptible.", "权力吸引可腐化之人。"),
    (
        "Governments always fail. They always fail because they are made of people.",
        "政府总会失败，因为它们由人组成。",
    ),
    ("The people who can destroy a thing, they control it.", "能毁灭某物的人，才是真正控制它的人。"),
    ("Lead them to paradise.", "带领他们走向天堂。"),
    (
        "Deep in the human unconscious is a pervasive need for a logical universe that makes sense. "
        "But the real universe is always one step beyond logic.",
        "人类潜意识深处有一种根深蒂固的需求--渴望一个合乎逻辑的宇宙。但真实的宇宙，永远超越逻辑一步。",
    ),
    ("Respect for the truth comes close to being the basis for all morality.", "对真相的尊重，几乎是一切道德的根基。"),
    (
        "The willow submits to the wind and prospers until one day it is many willows - a wall against the wind.",
        "柳树顺风而生，繁茂不止，终有一天柳成林--化作抵御风的墙。",
    ),
    ("Without change, something sleeps inside us, and seldom awakens.", "没有变化，内心某些东西便沉睡不醒。"),
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
