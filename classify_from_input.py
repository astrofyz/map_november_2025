"""
Utility script to classify street origins from user input.
Reads lines from stdin until EOF (Ctrl+D or Ctrl+Z).

Usage:
    python classify_from_input.py
"""

import sys
import re


def read_lines_until_eof_with_input():
    """
    Read lines from stdin until EOF (Ctrl+D or Ctrl+Z).
    
    Returns:
        List of strings, one per line
    """
    lines = []
    while True:
        try:
            line = input()
            lines.append(line)
        except EOFError:
            break
    return lines


def check_letters_in_order(text, letter_sequence):
    """
    Regex-based version: Check if letters appear in order using regular expressions.
    Creates a pattern like: letter1.*?letter2.*?letter3... that matches letters in sequence.
    """
    text = text.lower()
    letter_sequence = [c.lower() for c in letter_sequence]
    
    # Escape special regex characters in letters and join with .*? (non-greedy match)
    # This pattern will match letters in order with any characters (or none) in between
    escaped_letters = [re.escape(letter) for letter in letter_sequence]
    pattern = '.*?'.join(escaped_letters)
    
    # Search for the pattern in the text
    match = re.search(pattern, text, re.IGNORECASE)
    return match is not None


if __name__ == "__main__":
    all_tables = read_lines_until_eof_with_input()

    dangerous_letters = ['ш', 'а', 'ф', 'р', 'и', 'а', 'н']
    dangerous_letters1 = ['ш','а','ф','p','и','a','н']
    dangerous_letters2 = ['ш', 'а', 'ф', 'р', '1', 'а', 'н']
    
    
    variants = [
        ("dangerous_letters", dangerous_letters),
        ("dangerous_letters1", dangerous_letters1),
        ("dangerous_letters2", dangerous_letters2)
    ]
    
    for i, table in enumerate(all_tables):
        matches = []
        non_matches = []
        
        for variant_name, variant_sequence in variants:
            in_order = check_letters_in_order(table, variant_sequence)
            if in_order:
                print(i+1, table)
                # matches.append((variant_name, variant_sequence))
            # else:
                # non_matches.append((variant_name, variant_sequence))
        
        # # If any variant matches, print the result
        # if matches:
        #     print(f"{i+1}: {table[:80]}...")
        #     for variant_name, variant_sequence in matches:
        #         # print(f"    ✓ {variant_name}: Found all {info['found_count']}/{info['total_count']} letters in order")
        #         # print(f"      Indices: {indices}")
        #         print(f"      Letters: {variant_sequence}")
        #     print()
        
        # # Print debug info for non-matching variants if enabled
        # if SHOW_DEBUG_INFO and non_matches:
        #     if matches:
        #         print(f"    Debug - Non-matching variants:")
        #     for variant_name, variant_sequence, info in non_matches:
        #         print(f"      ✗ {variant_name}: Found {info['found_count']}/{info['total_count']} letters")
        #         if info['found']:
        #             print(f"        Found: {info['found']}")
        #         if info['missing']:
        #             print(f"        Missing: {[letter for _, letter in info['missing']]}")
        #         print()

