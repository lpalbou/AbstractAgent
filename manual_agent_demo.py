"""Manual script for the ReAct agent (not collected by pytest)."""

import sys
sys.path.insert(0, "src")

from abstractruntime import RunStatus
from abstractruntime.integrations.abstractcore import create_local_runtime

from abstractagent.agents.react import ReactAgent
from abstractagent.tools import ALL_TOOLS


def get_user_response(*, prompt: str, choices=None, allow_free_text: bool = True) -> str:
    """Minimal CLI prompt for handling ASK_USER waits (no external UI dependencies)."""
    prompt = str(prompt or "Please respond:").strip() or "Please respond:"
    if isinstance(choices, list) and choices:
        print(prompt)
        for i, c in enumerate(choices, start=1):
            print(f"  {i}) {c}")
        while True:
            raw = input("Select a number" + (" or type a response: " if allow_free_text else ": ")).strip()
            if allow_free_text and raw and not raw.isdigit():
                return raw
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= len(choices):
                    return str(choices[idx - 1])
            print("Invalid selection.")
    return input(f"{prompt} ").strip()


def on_step(step: str, data: dict) -> None:
    """Print step information."""
    if step == "init":
        print(f"Starting: {data.get('task', '')[:50]}")
    elif step == "reason":
        print(f"Reasoning (iteration {data.get('iteration', '?')})...")
    elif step == "parse":
        has_tools = data.get('has_tool_calls', False)
        if has_tools:
            print("Decided to use tools")
    elif step == "act":
        print(f"Tool: {data.get('tool', '')}({data.get('args', {})})")
    elif step == "observe":
        print(f"Result: {data.get('result', '')[:80]}...")
    elif step == "ask_user":
        print("Agent has a question...")
    elif step == "user_response":
        print(f"You answered: {data.get('response', '')[:50]}")
    elif step == "done":
        print(f"ANSWER: {data.get('answer', '')}")


def main():
    runtime = create_local_runtime(
        provider="ollama",
        model="qwen3:4b-instruct-2507-q4_K_M",
    )
    
    agent = ReactAgent(
        runtime=runtime,
        tools=ALL_TOOLS,
        on_step=on_step,
    )
    
    # Test task
    task = "List the files in the current directory"
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"{'='*60}\n")
    
    agent.start(task)
    
    # Run with question handling
    while True:
        state = agent.step()
        
        if state.status == RunStatus.COMPLETED:
            print(f"\n{'='*60}")
            print(f"Final Status: {state.status.value}")
            print(f"Iterations: {state.output.get('iterations', '?')}")
            print(f"Answer: {state.output.get('answer', 'N/A')}")
            print(f"{'='*60}\n")
            break
        
        elif state.status == RunStatus.WAITING:
            # Handle question
            question = agent.get_pending_question()
            if question:
                response = get_user_response(
                    prompt=question.get("prompt", "Please respond:"),
                    choices=question.get("choices"),
                    allow_free_text=question.get("allow_free_text", True),
                )
                agent.resume(response)
            else:
                print("Agent is waiting but no question found")
                break
        
        elif state.status == RunStatus.FAILED:
            print(f"Failed: {state.error}")
            break


if __name__ == "__main__":
    main()
