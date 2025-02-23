import pytest
from src.boyer_moore import boyer_moore_search

def test_basic_pattern_match():
    """Test basic pattern matching"""
    text = "ABAAABCD"
    pattern = "ABC"
    assert boyer_moore_search(text, pattern) == [4]

def test_multiple_occurrences():
    """Test finding multiple occurrences of a pattern"""
    text = "ABABABAB"
    pattern = "ABAB"
    assert boyer_moore_search(text, pattern) == [0, 2, 4]

def test_no_occurrences():
    """Test when pattern is not found"""
    text = "ABCDEF"
    pattern = "XYZ"
    assert boyer_moore_search(text, pattern) == []

def test_pattern_longer_than_text():
    """Test when pattern is longer than text"""
    text = "SHORT"
    pattern = "VERYLONGPATTERN"
    assert boyer_moore_search(text, pattern) == []

def test_empty_text():
    """Test searching in an empty text"""
    text = ""
    pattern = "ABC"
    assert boyer_moore_search(text, pattern) == []

def test_overlapping_patterns():
    """Test overlapping pattern matches"""
    text = "AAAAA"
    pattern = "AA"
    assert boyer_moore_search(text, pattern) == [0, 1, 2, 3]

def test_case_sensitive_match():
    """Test case sensitivity"""
    text = "abcABC"
    pattern = "abc"
    assert boyer_moore_search(text, pattern) == []
    pattern = "ABC"
    assert boyer_moore_search(text, pattern) == [3]

def test_error_handling_non_string_input():
    """Test error handling for non-string inputs"""
    with pytest.raises(TypeError):
        boyer_moore_search(123, "pattern")
    with pytest.raises(TypeError):
        boyer_moore_search("text", 456)

def test_error_handling_empty_pattern():
    """Test error handling for empty pattern"""
    with pytest.raises(ValueError):
        boyer_moore_search("text", "")