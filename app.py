# app.py (Gradio 6.6.0 - Fixed history handling, last 2 interactions)
import sys
import os
import json

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gradio as gr
from agent.agent import GithubProjectMentor, graph

# Initialize the agent globally
agent = GithubProjectMentor()

# Number of past interactions to include in context
MAX_HISTORY_INTERACTIONS = 2  # Each interaction = 1 user msg + 1 assistant msg


def save_graph():
    """Save the agent graph and return the path."""
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        graph_path = os.path.join(SRC_DIR, "agent_graph.png")
        with open(graph_path, "wb") as f:
            f.write(png_bytes)
        return graph_path
    except Exception:
        return None


def format_response(response) -> str:
    """
    Convert agent response to a Markdown string suitable for Gradio Chatbot.
    Handles: str, dict, list, or any other type.
    """
    # Already a string - return as-is
    if isinstance(response, str):
        return response
    
    # Handle dict response (structured output from LLM)
    if isinstance(response, dict):
        formatted_parts = []
        
        for key, value in response.items():
            # Format the key as a Markdown header
            header = f"## {key}"
            
            if isinstance(value, list):
                # Format list items as bullet points
                items = "\n".join([f"- {item}" for item in value])
                formatted_parts.append(f"{header}\n{items}")
            elif isinstance(value, dict):
                # Nested dict - format as key-value pairs
                nested = "\n".join([f"- **{k}:** {v}" for k, v in value.items()])
                formatted_parts.append(f"{header}\n{nested}")
            else:
                formatted_parts.append(f"{header}\n{value}")
        
        return "\n\n".join(formatted_parts)
    
    # Handle list response
    if isinstance(response, list):
        return "\n".join([f"- {item}" for item in response])
    
    # Fallback: convert to string
    return str(response)


def ensure_string(value) -> str:
    """Ensure a value is a string, converting if necessary."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return format_response(value)
    if isinstance(value, list):
        return "\n".join([f"- {item}" for item in value])
    return str(value)


def respond(message, chat_history):
    """Generate response and update chat history."""
    if not message.strip():
        return "", chat_history
    
    # ═══════════════════════════════════════════════════════════════════
    # Only use the last N interactions for agent context
    # Each interaction = 2 messages (user + assistant)
    # ═══════════════════════════════════════════════════════════════════
    max_messages = MAX_HISTORY_INTERACTIONS * 2
    recent_history = chat_history[-max_messages:] if len(chat_history) > max_messages else chat_history
    
    # Convert Gradio history (list of dicts) to your agent's format
    agent_history = []
    for msg in recent_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Ensure content is always a string
        content = ensure_string(content)
        
        # Map "assistant" to "agent" for your agent's format
        agent_role = "agent" if role == "assistant" else "user"
        agent_history.append({"role": agent_role, "content": content})
    
    try:
        response = agent.query(message, history=agent_history)
        # Format response to ensure it's always a string
        formatted_response = format_response(response)
    except Exception as e:
        formatted_response = f"❌ **Error:** {str(e)}"
    
    # Append new messages in Gradio 6.x format: {"role": "...", "content": "..."}
    # (Full history is kept in UI, only last N sent to agent)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": formatted_response})
    
    return "", chat_history


def clear_chat():
    """Clear the chat history."""
    return [], ""


# Build the interface using gr.Blocks
with gr.Blocks(title="GitHub Project Mentor") as demo:
    
    # Header
    gr.Markdown("""
    # 🤖 GitHub Project Mentor Agent
    Your AI assistant for **project guidance**, **GitHub integration**, and **knowledge retrieval**.
    """)
    
    # Chatbot with Gradio 6.x parameters
    chatbot = gr.Chatbot(
        height=500,
        placeholder="Ask me anything about your project...",
        layout="bubble",  # or "panel"
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your message here...",
            label="Your Message",
            scale=9,
        )
        submit_btn = gr.Button("Send 📤", scale=1, variant="primary")
    
    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear Chat")
    
    # Example prompts
    gr.Markdown("### 💡 Try these examples:")
    gr.Examples(
        examples=[
            "What can you help me with?",
            "Explain the project structure",
            "How do I create a new GitHub repository?",
            "What are best practices for Python projects?",
        ],
        inputs=msg,
    )
    
    # Event handlers
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(clear_chat, outputs=[chatbot, msg])
    
    # Footer
    gr.Markdown("---\n*Built with ❤️ using LangGraph & Gradio*")


if __name__ == "__main__":
    """
    # Save the graph on startup
    graph_path = save_graph()
    if graph_path:
        print(f"✅ Graph saved to: {graph_path}")
    """

    print("\n🚀 Starting Gradio interface at http://localhost:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )