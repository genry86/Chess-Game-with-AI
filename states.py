"""
Chess game state definitions.
Contains TypedDict classes for representing game state, AI state, and possible moves.
"""

from typing import TypedDict, Annotated, List, Literal
import operator

class AIState(TypedDict):
    """State for AI decision-making process."""
    board_state: dict
    new_step: str
    new_fen: str

class GameState(TypedDict):
    """Main game state containing all game information."""
    user_color: str  # 'white' or 'black'
    ai_color: str  # 'white' or 'black'
    turn: str  # 'user' or 'ai'
    status: str  # 'playing', 'mate', 'draw', 'resignation'
    fen: str
    history_fens: Annotated[List[str], operator.add]
    ai_state: AIState