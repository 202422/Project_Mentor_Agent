# ask_agent.py
import sys
import os

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from agent.agent import GithubProjectMentor, graph

console = Console()


def display_graph():
    """Render and save the LangGraph agent graph."""
    try:
        png_bytes  = graph.get_graph().draw_mermaid_png()
        graph_path = os.path.join(SRC_DIR, "agent_graph.png")

        with open(graph_path, "wb") as f:
            f.write(png_bytes)

        console.print(f"[dim]Graph saved to: {graph_path}[/dim]\n")

    except Exception as e:
        console.print(f"[yellow][Could not render graph: {e}][/yellow]")
        console.print("[dim][Hint: pip install playwright && playwright install chromium][/dim]\n")


def main():
    agent   = GithubProjectMentor()
    history = []

    display_graph()

    console.print(Rule("[bold cyan]Agent Query Interface[/bold cyan]"))
    console.print("[dim]Type [bold]exit[/bold] to quit.[/dim]\n")

    while True:
        user_input = console.input("[bold green]You:[/bold green] ").strip()

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            console.print("\n[bold cyan]Goodbye![/bold cyan]")
            break

        try:
            response = agent.query(user_input, history=history)

            console.print()
            console.print(
                Panel(
                    Markdown(response),
                    title="[bold cyan]Agent[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
            console.print()

            history.append({"role": "user",  "content": user_input})
            history.append({"role": "agent", "content": response})

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}\n")


if __name__ == "__main__":
    main()