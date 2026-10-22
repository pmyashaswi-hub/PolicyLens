"""
PolicyLens Conversation Memory Module
Maintains multi-turn dialog history for contextual query rewriting and follow-ups.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from src.config import MAX_MEMORY_TURNS


@dataclass
class ChatTurn:
    """Represents a single conversational interaction turn."""
    role: str  # 'user' or 'assistant'
    content: str
    citations: List[int] = field(default_factory=list)
    found: bool = True
    confidence: str = "high"
    missing_info: Optional[str] = None


class ConversationMemory:
    """
    In-memory FIFO buffer retaining the last N conversation turns.
    Integrates directly with Streamlit session state.
    """

    def __init__(self, max_turns: int = MAX_MEMORY_TURNS):
        self.max_turns = max_turns
        self.history: List[ChatTurn] = []

    def add_user_message(self, message: str) -> None:
        """Record user query."""
        self.history.append(ChatTurn(role="user", content=message.strip()))
        self._trim()

    def add_assistant_message(
        self,
        content: str,
        citations: Optional[List[int]] = None,
        found: bool = True,
        confidence: str = "high",
        missing_info: Optional[str] = None
    ) -> None:
        """Record assistant response along with grounding metadata."""
        self.history.append(
            ChatTurn(
                role="assistant",
                content=content.strip(),
                citations=citations or [],
                found=found,
                confidence=confidence,
                missing_info=missing_info
            )
        )
        self._trim()

    def _trim(self) -> None:
        """Retain only the most recent max_turns."""
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_recent_turns(self, n: int = 4) -> List[Dict[str, Any]]:
        """Return the last n turns formatted as dictionaries."""
        turns = self.history[-(n * 2):] if len(self.history) > (n * 2) else self.history
        return [asdict(t) for t in turns]

    def format_history_for_prompt(self, n: int = 4) -> str:
        """Format the last n turns into a compact text block for LLM prompts."""
        recent = self.get_recent_turns(n=n)
        if not recent:
            return "No previous conversation history."
        lines = []
        for turn in recent:
            role_label = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {turn['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset conversation memory."""
        self.history = []

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Serialize memory for session state storage."""
        return [asdict(t) for t in self.history]

    @classmethod
    def from_dict_list(cls, data: List[Dict[str, Any]], max_turns: int = MAX_MEMORY_TURNS) -> "ConversationMemory":
        """Instantiate memory from serialized session state list."""
        memory = cls(max_turns=max_turns)
        for item in data:
            memory.history.append(
                ChatTurn(
                    role=item.get("role", "user"),
                    content=item.get("content", ""),
                    citations=item.get("citations", []),
                    found=item.get("found", True),
                    confidence=item.get("confidence", "high"),
                    missing_info=item.get("missing_info")
                )
            )
        memory._trim()
        return memory
