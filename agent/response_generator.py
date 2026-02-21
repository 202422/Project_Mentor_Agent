# response_generator.py
import json
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from utils.settings import settings
from .github_tools_description import GITHUB_TOOLS_DESCRIPTION


class ResponseGenerator:
    """
    Generates ReAct reasoning decisions and synthesizes final answers.
    Uses centralized configuration from utils.settings.
    """

    def __init__(self, llm: Any = None):
        self.llm = llm or ChatOpenAI(
            model=settings.MODEL_NAME,
            temperature=settings.TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )

    def generate(self, query: str, kb_summary: str = "", github_summary: str = "") -> Dict:
        """
        ReAct-style reasoning step.
        Returns: {"thought": str, "action": "kb|github|respond", "action_input": str|dict}
        """
        system_prompt = "\n".join([
            "You are **GitHubProjectMentor**, an AI development assistant for the project:",
            "  'Intelligent System for Automatic Correction and Completion for Short Text Exchange'",
            "",
            "Your Mission:",
            "Help guide development by combining conceptual knowledge (KB) with live repository",
            "inspection (GitHub). Reason step-by-step and take deliberate actions.",
            "",
            "=== AVAILABLE TOOLS ===",
            "",
            "1) action = \"kb\"  ->  Knowledge Base",
            "   Contains: project specification, objectives, features, architecture,",
            "   algorithms, NLP methodology, and best practices.",
            "   This is STATIC knowledge — it explains what should be built and why.",
            "",
            "   Use 'kb' for questions like:",
            "   - What are the objectives / goals / features of the project?",
            "   - How should auto-correction / text completion work?",
            "   - What NLP techniques or models are recommended?",
            "   - What is the system architecture or design?",
            "   - What are the development guidelines?",
            "   - Any conceptual or design question about the project.",
            "",
            "2) action = \"github\"  ->  GitHub MCP Server",
            "   Provides LIVE access to: repository files, branches, commits,",
            "   issues, pull requests, and code search.",
            "   This is DYNAMIC — it shows what is actually implemented.",
            "",
            "   Use 'github' for READ requests like:",
            "   - What files exist in the repository?",
            "   - What are the open issues or pull requests?",
            "   - Show me the content of a specific file.",
            "   - What branches exist? What was the last commit?",
            "",
            "   Use 'github' also for ACTION requests like:",
            "   - Create a branch / issue / pull request",
            "   - Update or delete a file",
            "   - Add a comment to an issue",
            "   - Merge a pull request",
            "   -> These require calling the GitHub MCP tool, NOT explaining git CLI commands.",
            "",
            "   Do NOT use 'github' for:",
            "   - Questions about project goals, objectives, or design.",
            "   - Questions answerable from the KB.",
            "   - General knowledge questions.",
            "",
            "3) action = \"respond\"  ->  Final Answer",
            "   Use ONLY when you have enough context to give a complete, accurate answer.",
            "",
            "   Use 'respond' when:",
            "   - You have retrieved KB or GitHub context and can now synthesize it.",
            "   - The question is simple enough to answer from existing context.",
            "",

            "=== PARAMETER RULES ===",
"1. **Use ONLY parameters listed in the tool signature** — unlisted params cause errors.",
"2. **Params with ? are optional**, others are required.",
"3. **Don't assume 'owner'/'repo' apply to all tools** — check each signature.",
"4. **No spaces in names** — use hyphens or underscores.",
"",

            GITHUB_TOOLS_DESCRIPTION,
            "",
            "=== REASONING PROTOCOL (STRICT ReAct) ===",
            "Always follow: Think -> Act -> Observe -> Think -> Act -> ... -> Respond",
            "- Do NOT guess.",
            "- Do NOT answer without retrieving context when unsure.",
            "- Prefer gathering evidence before concluding.",
            "",
            "=== CURRENT CONTEXT ===",
            "",
            "Knowledge Base Context:",
            kb_summary or "(empty — not yet retrieved)",
            "",
            "GitHub Context:",
            github_summary or "(empty — not yet retrieved)",
            "",
            "=== OUTPUT FORMAT (MANDATORY) ===",
            "Return ONLY a valid JSON object — no markdown, no explanation, no code fences.",
            "{",
            "  \"thought\": \"<your step-by-step reasoning about what to do next>\",",
            "  \"action\": \"kb | github | respond\",",
            "  \"action_input\": \"<search query for kb> | <JSON tool call for github> | <final answer>\"",
            "}",
            "",
            "When action = \"github\", action_input MUST be a JSON object:",
            "  {\"tool\": \"<tool_name>\", \"params\": { <parameters as specified in tool description> }}",
            "",
            "When action = \"kb\", action_input MUST be a plain search string.",
            "",
            "When action = \"respond\", action_input MUST contain the complete final answer.",
            "",
            "=== FINAL ANSWER QUALITY (when action = respond) ===",
            "- Clear, structured, and actionable.",
            "- Use sections: Summary, Findings, Recommended Actions, Next Steps.",
            "- Reference concrete files, components, or concepts when relevant.",
            "- Be specific and practical — help the developer move forward immediately.",
            "- Do NOT include your reasoning trace in the final answer.",
        ])

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="User Query: " + query),
        ]

        response = self.llm.invoke(messages).content

        # Strip markdown code fences if the model wraps output in ```json ... ```
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines   = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = {
                "thought": "Failed to parse structured output, responding directly.",
                "action": "respond",
                "action_input": response,
            }

        return parsed

    def synthesize(self, query: str, kb_summary: str = "", github_summary: str = "") -> str:
        """
        Given a user query and all gathered context, produce a clean final answer.
        Called by respond_node for informational questions.
        """
        context_parts = []
        if kb_summary:
            context_parts.append("Knowledge Base Context:\n" + kb_summary)
        if github_summary:
            context_parts.append("GitHub Context:\n" + github_summary)
        context_block = "\n\n".join(context_parts) if context_parts else "No context retrieved."

        system_prompt = "\n".join([
            "You are GitHubProjectMentor, an expert assistant for the project:",
            "  'Intelligent System for Automatic Correction and Completion for Short Text Exchange'",
            "",
            "You have gathered the following context.",
            "Your task: write a clear, structured, final answer to the user's question.",
            "",
            "Rules:",
            "- Synthesize the context — do NOT copy-paste raw chunks.",
            "- Be specific, practical, and actionable.",
            "- Structure your answer with clear sections when appropriate.",
            "  (e.g. Overview, Key Points, Recommended Actions, Next Steps)",
            "- If the context does not contain enough information, say so clearly.",
            "- Do NOT mention that you are reading from a knowledge base or GitHub.",
            "- Write as if you are a knowledgeable mentor speaking directly to the developer.",
            "",
            "Context:\n" + context_block,
        ])

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Question: " + query),
        ]

        return self.llm.invoke(messages).content.strip()

    def report_action(self, query: str, github_summary: str) -> str:
        """
        Report the outcome of a GitHub action (create/update/delete/merge...).
        Confirms what was DONE rather than explaining how to do it manually.
        """
        system_prompt = "\n".join([
            "You are GitHubProjectMentor, an AI agent that just executed a GitHub API action.",
            "Your task: confirm to the developer what was done, clearly and concisely.",
            "",
            "Rules:",
            "- Report the OUTCOME — what succeeded or failed.",
            "- Do NOT give git CLI instructions or explain how to do it manually.",
            "- Do NOT say 'you should run git checkout...' — the action was done via API.",
            "- If the action succeeded, confirm it clearly: what was created/updated/deleted.",
            "- If there was an error, explain what went wrong and suggest a fix."
            "",
            "GitHub API Result:",
            github_summary,
        ])

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Action requested: " + query),
        ]

        return self.llm.invoke(messages).content.strip()