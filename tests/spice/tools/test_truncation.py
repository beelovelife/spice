from spice.tools.base import truncate_head_tail


def test_head_tail_truncation_keeps_both_ends() -> None:
    text = "HEAD" + "x" * 20 + "TAIL"

    result = truncate_head_tail(text, 10)

    assert result.startswith("HEADx")
    assert result.endswith("xTAIL")
    assert "omitted 18 characters" in result
    assert "kept first 5 and last 5" in result


def test_head_tail_truncation_leaves_short_text_unchanged() -> None:
    assert truncate_head_tail("short", 10) == "short"

