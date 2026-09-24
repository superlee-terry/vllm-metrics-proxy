"""Test loop detection logic — chunk-level repetition."""
import sys
sys.path.insert(0, "/mnt/data/vllm-metrics-proxy")

from vllm_metrics_proxy.proxy import _detect_loop, _punctuation_spam
from vllm_metrics_proxy.config import settings


def test_tail_match():
    """Strategy 1: last 5 chunks identical -> loop detected.

    Note: chunks must satisfy the per-chunk floor (>=5 chars, >=2 distinct
    chars) — a 3-char chunk can never match.
    """
    window = ["hello wo"] * 5
    result, reason = _detect_loop(window, repeat_threshold=3, min_tail_match=5)
    assert result is True, f"Expected True, got {result}"
    assert "tail_match" in reason, f"Reason should contain tail_match: {reason}"
    print(f"  PASS: tail_match -> {reason[:80]}")


def test_chunk_repeat_dense():
    """Strategy 2: a long chunk repeated densely (high count, small gaps) -> loop.

    The detector uses a *density* heuristic: a >=min_len chunk that appears
    many times with small gaps (a real loop) triggers, while the same chunk
    sprinkled sparsely through a long window (e.g. enumeration) does not.
    """
    chunk = "这是一个重复的输出片段"
    window = ["a", chunk, chunk, chunk, chunk, "b"]  # 4 dense occurrences
    result, reason = _detect_loop(window, repeat_threshold=3, min_tail_match=5)
    assert result is True, f"Expected True, got {result}, reason={reason}"
    assert "chunk_repeat" in reason, f"Reason should contain chunk_repeat: {reason}"
    print(f"  PASS: chunk_repeat dense -> {reason[:80]}")


def test_chunk_repeat_sparse_not_triggered():
    """Same long chunk, but sprinkled sparsely -> NOT a loop (enumeration)."""
    chunk = "这是一个重复的输出片段"
    window = [
        "intro", chunk, "o1", "o2", "o3", "o4", "o5", "o6", "o7", chunk,
        "o8", "o9", "o10", "o11", "o12", "o13", "o14", "o15", "o16", "o17", chunk,
    ]
    result, reason = _detect_loop(window, repeat_threshold=3, min_tail_match=5)
    assert result is False, f"Sparsely-repeated chunk should not trigger: {reason}"
    print(f"  PASS: chunk_repeat sparse -> no loop")


def test_no_loop_diverse():
    """No loop when chunks are diverse."""
    window = [f"chunk_{i}_content_xyz_long_enough_20chars" for i in range(15)]
    result, reason = _detect_loop(window, repeat_threshold=3, min_tail_match=5)
    assert result is False, f"Expected False, got {result} reason={reason}"
    print(f"  PASS: no_loop -> {reason}")


def test_window_too_small():
    """Should not trigger when window < min_tail_match."""
    window = ["abc"] * 3
    result, reason = _detect_loop(window, repeat_threshold=3, min_tail_match=5)
    assert result is False, f"Expected False for small window, got {result}"
    print(f"  PASS: window_too_small -> {reason}")


def test_short_chunks_not_triggering():
    """Short chunks (< min_len) should NOT trigger tail_match.

    A 2-char chunk can never satisfy the >=5-char per-chunk floor, so even
    15 identical ones must NOT be flagged (prevents false loops on tiny
    token-level chunks like 'ab', '、', etc.).
    """
    window = ["ab"] * 15
    result, reason = _detect_loop(window, repeat_threshold=3, min_tail_match=5)
    assert result is False, f"2-char chunks should not trigger tail_match: {reason}"
    print(f"  PASS: short_chunks_not_triggered -> no loop")


def test_thinking_like_pattern_not_triggering():
    """Simulate thinking patterns — should NOT trigger."""
    window = [
        "让我们逐步分析这个问题。",
        "首先，我们需要理解",
        "接下来考虑另一种情况",
        "然后我们可以得出结论",
        "最后总结一下上面的分析",
        "这个问题的关键在于",
        "从另一个角度来看",
        "综上所述，答案是",
        "让我们回顾一下步骤",
        "最终结果是正确的",
    ]
    result, reason = _detect_loop(window, repeat_threshold=3, min_tail_match=5)
    assert result is False, f"Thinking-like patterns should not trigger loop: {reason}"
    print(f"  PASS: thinking_pattern -> no loop detected")


def test_similar_but_different_chunks():
    """Chunks that are similar but not identical should NOT trigger."""
    # Each chunk is long enough but differs slightly
    window = [
        "重复内容版本A，这段文字足够长超过20个字符以上",
        "重复内容版本B，这段文字足够长超过20个字符以上",
        "重复内容版本C，这段文字足够长超过20个字符以上",
        "filler_1",
        "filler_2",
        "filler_3",
        "filler_4",
        "filler_5",
        "filler_6",
        "filler_7",
    ]
    result, reason = _detect_loop(window, repeat_threshold=3, min_tail_match=5)
    assert result is False, f"Similar but different chunks should not trigger: {reason}"
    print(f"  PASS: similar_different -> no loop detected")


def test_chunk_repeat_boundary():
    """Exactly 2 repetitions (below threshold=3) should NOT trigger."""
    chunk = "这是一个刚好10个字符以上的测试内容"
    window = [
        "intro",
        chunk,
        "other_1",
        chunk,
        "other_2",
        "other_3",
        "other_4",
        "other_5",
        "other_6",
        "other_7",
    ]
    result, reason = _detect_loop(window, repeat_threshold=3, min_tail_match=5)
    assert result is False, f"2 repetitions should not trigger (threshold=3): {reason}"
    print(f"  PASS: chunk_repeat_boundary -> no loop (only 2x)")


def test_real_world_hello_world_loop():
    """The actual case that triggered before: '世界\n你好' repeated.
    Now it should NOT trigger because individual chunks are short (<10 chars)."""
    # Simulate the actual pattern from the log
    window = [
        "世界",
        "\n你好",
        "世界",
        "\n你好",
        "世界",
        "\n你好",
        "世界",
        "\n你好",
        "世界",
        "\n你好",
    ]
    result, reason = _detect_loop(window, repeat_threshold=3, min_tail_match=5)
    # Individual chunks are too short (<10 chars) for chunk_repeat
    # tail_match won't trigger because chunks alternate between 2 values
    assert result is False, f"Short alternating chunks should not trigger: {reason}"
    print(f"  PASS: hello_world_alternating -> no false positive")


def test_punctuation_spam():
    """Trailing pure-punctuation runs (user's examples) must be caught."""
    min_chars = settings.loop_punct_spam_min_chars  # 10
    min_chunks = settings.loop_punct_spam_min_chunks  # 6
    cases = {
        "!!!!! x3": ["!!!!!"] * 3,
        "...... x2": ["......"] * 2,
        "、 x8 (one token/chunk)": ["、"] * 8,
        "、 x6 (chunk threshold)": ["、"] * 6,
        "mixed CN/EN": ["！！！", "...", "！？"] * 2,
    }
    for name, window in cases.items():
        result, reason = _punctuation_spam(window, min_chars, min_chunks)
        assert result is True, f"{name} should trigger: {reason}"
        print(f"  PASS: {name} -> {reason}")


def test_punctuation_spam_not_triggered():
    """Single ellipsis/emphasis and structural symbols must NOT be cut."""
    min_chars = settings.loop_punct_spam_min_chars  # 10
    min_chunks = settings.loop_punct_spam_min_chunks  # 6
    cases = {
        "single ......": ["......"],
        "single !!!!!": ["!!!!!"],
        "、 x3 (short run)": ["、"] * 3,
        "table border ─": ["──────"] * 3,
        "box ┌────┐": ["┌────┐"] * 3,
        "markdown ###": ["###", "###", "###"],
        "dash ----": ["----"] * 4,
        "em-dash ——": ["——"] * 4,
        "punct then letter": ["!!a!!", "!!a!!", "!!a!!"],
    }
    for name, window in cases.items():
        result, reason = _punctuation_spam(window, min_chars, min_chunks)
        assert result is False, f"{name} should NOT trigger: {reason}"
        print(f"  PASS: {name} -> not triggered")


if __name__ == "__main__":
    print("Running loop detection tests...\n")
    test_tail_match()
    test_chunk_repeat_dense()
    test_chunk_repeat_sparse_not_triggered()
    test_no_loop_diverse()
    test_window_too_small()
    test_short_chunks_not_triggering()
    test_thinking_like_pattern_not_triggering()
    test_similar_but_different_chunks()
    test_chunk_repeat_boundary()
    test_real_world_hello_world_loop()
    test_punctuation_spam()
    test_punctuation_spam_not_triggered()
    print("\nAll tests passed!")
