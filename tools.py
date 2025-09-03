from pydantic import BaseModel, Field
from langchain_core.tools import tool
from utility import is_legal_step, fen_to_json, apply_uci_move

@tool
def check_step_is_legal(
    fen: str = Field(..., description="Current board FEN"),
    step: str = Field(..., description="Candidate move in UCI format")
) -> bool:
    """
    Validate whether UCI move `step` is legal for the current board `fen`.
    """
    is_legal = is_legal_step(fen=fen, uci=step)
    return is_legal

@tool
def get_new_board_state(
    fen: str = Field(..., description="Current board FEN"),
    step: str = Field(..., description="Candidate move in UCI format"),
    opponent_color: str = Field(..., description="Color of the opponent side")
):
    """
    Fetch new board json that describes all pieces and their possible steps.
    `fen` - current board fen,
    `step` new piece step,
    `opponent_color` - color of opponent player.
    """
    is_legal = is_legal_step(fen=fen, uci=step)
    if is_legal:
        new_fen = apply_uci_move(fen=fen, uci=step)
        board_json = fen_to_json(fen=new_fen)
    else:
        board_json = fen_to_json(fen=fen)

    opponent_board_json = board_json[opponent_color]
    return opponent_board_json