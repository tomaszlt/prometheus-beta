def to_sentence_case(text: str) -> str:
    """
    Convert a string to sentence case.
    
    Args:
        text (str): The input string to convert.
    
    Returns:
        str: The string converted to sentence case.
    
    Raises:
        TypeError: If input is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    
    if not text:
        return ""
    
    return text[0].upper() + text[1:].lower()