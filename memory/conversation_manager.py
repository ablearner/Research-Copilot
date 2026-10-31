"""Compatibility import for conversation language detection.

Conversation lifecycle now lives in services.research.conversation_service.
"""
from services.research.conversation_service import _preferred_answer_language_from_text

__all__ = ["_preferred_answer_language_from_text"]
