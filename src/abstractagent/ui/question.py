"""Interactive question UI for agent-user interaction.

Provides a clean terminal UI for:
- Multiple choice questions
- Free text input
- Combined choice + free text
"""

import sys
from typing import List, Optional, Dict, Any


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    WHITE = "\033[37m"
    
    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"


def _supports_color() -> bool:
    """Check if terminal supports colors."""
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    return True


def _c(text: str, *codes: str) -> str:
    """Apply color codes if supported."""
    if not _supports_color():
        return text
    return "".join(codes) + text + Colors.RESET


class QuestionUI:
    """Interactive question UI component."""
    
    def __init__(
        self,
        prompt: str,
        choices: Optional[List[str]] = None,
        allow_free_text: bool = True,
    ):
        self.prompt = prompt
        self.choices = choices or []
        self.allow_free_text = allow_free_text
    
    def render(self) -> None:
        """Render the question UI to terminal."""
        print()
        print(_c("┌" + "─" * 58 + "┐", Colors.CYAN))
        print(_c("│", Colors.CYAN) + _c(" 🤖 Agent Question", Colors.BOLD, Colors.YELLOW) + " " * 39 + _c("│", Colors.CYAN))
        print(_c("├" + "─" * 58 + "┤", Colors.CYAN))
        
        # Word wrap the prompt
        words = self.prompt.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= 56:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        for line in lines:
            padding = 56 - len(line)
            print(_c("│", Colors.CYAN) + f" {line}" + " " * padding + _c("│", Colors.CYAN))
        
        print(_c("├" + "─" * 58 + "┤", Colors.CYAN))
        
        # Render choices if any
        if self.choices:
            print(_c("│", Colors.CYAN) + _c(" Options:", Colors.DIM) + " " * 48 + _c("│", Colors.CYAN))
            for i, choice in enumerate(self.choices, 1):
                choice_text = f"  [{i}] {choice}"
                if len(choice_text) > 56:
                    choice_text = choice_text[:53] + "..."
                padding = 56 - len(choice_text)
                print(_c("│", Colors.CYAN) + _c(f" [{i}]", Colors.GREEN, Colors.BOLD) + f" {choice}" + " " * (padding - 4) + _c("│", Colors.CYAN))
            
            if self.allow_free_text:
                print(_c("│", Colors.CYAN) + " " * 57 + _c("│", Colors.CYAN))
                free_text = f"  [0] Type your own response..."
                padding = 56 - len(free_text)
                print(_c("│", Colors.CYAN) + _c("  [0]", Colors.MAGENTA, Colors.BOLD) + " Type your own response..." + " " * (padding - 5) + _c("│", Colors.CYAN))
        
        print(_c("└" + "─" * 58 + "┘", Colors.CYAN))
        print()
    
    def get_input(self) -> str:
        """Get user input with validation."""
        while True:
            if self.choices:
                prompt_text = _c("Enter choice (1-" + str(len(self.choices)) + ") or 0 for custom: ", Colors.CYAN)
            else:
                prompt_text = _c("Your response: ", Colors.CYAN)
            
            try:
                user_input = input(prompt_text).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return ""
            
            if not user_input:
                continue
            
            # Handle choice selection
            if self.choices:
                try:
                    choice_num = int(user_input)
                    if choice_num == 0 and self.allow_free_text:
                        # Free text mode
                        print()
                        custom_input = input(_c("Type your response: ", Colors.MAGENTA)).strip()
                        if custom_input:
                            return custom_input
                        continue
                    elif 1 <= choice_num <= len(self.choices):
                        return self.choices[choice_num - 1]
                    else:
                        print(_c(f"Please enter 1-{len(self.choices)}" + (" or 0" if self.allow_free_text else ""), Colors.YELLOW))
                        continue
                except ValueError:
                    # Not a number - treat as free text if allowed
                    if self.allow_free_text:
                        return user_input
                    print(_c("Please enter a number.", Colors.YELLOW))
                    continue
            else:
                # No choices - just return the input
                return user_input
    
    def ask(self) -> str:
        """Render and get response in one call."""
        self.render()
        return self.get_input()


def render_question(
    prompt: str,
    choices: Optional[List[str]] = None,
    allow_free_text: bool = True,
) -> None:
    """Render a question UI without getting input."""
    ui = QuestionUI(prompt, choices, allow_free_text)
    ui.render()


def get_user_response(
    prompt: str,
    choices: Optional[List[str]] = None,
    allow_free_text: bool = True,
) -> str:
    """Show question UI and get user response.
    
    Args:
        prompt: The question to ask
        choices: Optional list of choices
        allow_free_text: Whether to allow free text input (always True for option 0)
    
    Returns:
        User's response string
    """
    ui = QuestionUI(prompt, choices, allow_free_text)
    return ui.ask()


# Async version for use with asyncio
async def get_user_response_async(
    prompt: str,
    choices: Optional[List[str]] = None,
    allow_free_text: bool = True,
) -> str:
    """Async version of get_user_response."""
    import asyncio
    
    ui = QuestionUI(prompt, choices, allow_free_text)
    ui.render()
    
    # Run input in executor to not block event loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, ui.get_input)
