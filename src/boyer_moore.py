def boyer_moore_search(text, pattern):
    """
    Implement the Boyer-Moore string search algorithm.
    
    Args:
        text (str): The text to search in
        pattern (str): The pattern to search for
    
    Returns:
        list: A list of starting indices where the pattern is found in the text
    
    Raises:
        TypeError: If inputs are not strings
        ValueError: If pattern is empty
    """
    # Input validation
    if not isinstance(text, str) or not isinstance(pattern, str):
        raise TypeError("Both text and pattern must be strings")
    
    if not pattern:
        raise ValueError("Pattern cannot be empty")
    
    # Bad character heuristic preprocessing
    def preprocess_bad_character(pattern):
        bad_char = {}
        for i in range(len(pattern)):
            bad_char[pattern[i]] = i
        return bad_char
    
    # Good suffix heuristic preprocessing
    def preprocess_good_suffix(pattern):
        m = len(pattern)
        good_suffix = [0] * m
        suffix = [0] * m
        
        # Compute suffix array
        suffix[m-1] = m
        for i in range(m-2, -1, -1):
            j = 0
            while j < m-1-i and pattern[m-1-j] == pattern[m-1-j-i-1]:
                j += 1
            suffix[i] = j
        
        # Compute good suffix shift
        for i in range(m):
            good_suffix[i] = m
        
        j = 0
        for i in range(m-1, -1, -1):
            if suffix[i] == m-i-1:
                while j < m-1-i:
                    if good_suffix[j] == m:
                        good_suffix[j] = m-1-i
                    j += 1
        
        for i in range(m):
            good_suffix[m-1-suffix[i]] = m-1-i
        
        return good_suffix
    
    # Main search algorithm
    def search(text, pattern, bad_char, good_suffix):
        results = []
        m, n = len(pattern), len(text)
        
        # Preprocessing
        if m > n:
            return results
        
        i = 0  # Text index
        while i <= n - m:
            j = m - 1  # Pattern index
            
            # Try to match pattern from right to left
            while j >= 0 and pattern[j] == text[i+j]:
                j -= 1
            
            # If pattern is found
            if j < 0:
                results.append(i)
                # Shift based on good suffix or bad character
                i += m - good_suffix[0] if i + m < n else 1
            else:
                # Compute shifts
                bad_char_shift = j - bad_char.get(text[i+j], -1)
                good_suffix_shift = good_suffix[j]
                
                # Take the maximum shift
                i += max(bad_char_shift, good_suffix_shift)
        
        return results

    # Preprocess pattern
    bad_char = preprocess_bad_character(pattern)
    good_suffix = preprocess_good_suffix(pattern)
    
    # Perform search
    return search(text, pattern, bad_char, good_suffix)