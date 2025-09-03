"""
Chess game logic implementation using LangGraph framework with RAG system for AI moves.
This module defines the game state and implements a stateful graph for chess gameplay.
"""
 
from typing import Dict, Any, Literal
import chess
from dotenv import load_dotenv

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from langchain_openai import ChatOpenAI
from langgraph.types import Command, interrupt

from utility import (
    is_valid_fen,
    is_legal_fen_transition,
    is_legal_step,
    find_possible_steps,
    fen_to_json,
    apply_uci_move,
    read_file,
    extract_final_move,
    load_rag,
    display_graph
)
from tools import check_step_is_legal, get_new_board_state
from states import GameState
from constants import ConfigConstants, ChessConstants, AIConstants, FilePathConstants, GraphNode

load_dotenv()
checkpointer = None

human_prompt = None
chess_rules = None

# Helper functions
def init_globals():
    global checkpointer, human_prompt, chess_rules

    checkpointer = InMemorySaver()

    if human_prompt is None:
        human_prompt = read_file(FilePathConstants.PROMPT_AGENT_START)

    if chess_rules is None:
        chess_rules = read_file(FilePathConstants.PROMPT_CHESS_RULES)

    load_rag()

def create_agent(tools: list, system_prompt: str):
    llm = ChatOpenAI(model="gpt-4.1", temperature=AIConstants.TEMPERATURE)
    llm = llm.bind_tools(tools)

    memory = MemorySaver()
    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        prompt=system_prompt
    ).with_retry(stop_after_attempt=AIConstants.MAX_RETRY_ATTEMPTS)
    return agent_executor

def create_agent_config(node_name: str) -> RunnableConfig:
    config: RunnableConfig = {
        "configurable": {"thread_id": f"chess_{node_name}"},
        "recursion_limit": AIConstants.RECURSION_LIMIT
    }
    return config

def get_step_from_last_response_message(response):
    new_step = "----"
    if "messages" in response and response["messages"]:
        last_message = response["messages"][-1]
        message_content = last_message.content.strip() if hasattr(last_message, 'content') else str(
            last_message).strip()
        new_step = extract_final_move(message_content)
    return new_step

def get_possible_steps(fen: str, ai_color: str) -> str:
    possible_steps = find_possible_steps(fen=fen, ai_color=ai_color)
    possible_steps_formated = []
    for step in possible_steps:
        possible_steps_formated.append(f"Next move: {step['next_move']}\n"
                                       f"Score: {step['score']}")
    possible_steps_text = "\n------\n".join(possible_steps_formated)
    return possible_steps_text

# Main graph Nodes
def start_node(state: GameState) -> GameState:
    """
    Initialize the game state with user's color choice.
    Sets up initial chess position and determines who goes first.
    """
    user_color: str = state["user_color"]
    turn: str = ChessConstants.TURN_USER if user_color == ChessConstants.COLOR_WHITE else ChessConstants.TURN_AI
    return {
        "user_color": state["user_color"],
        "ai_color": state["ai_color"],
        "turn": turn,
        "status": ChessConstants.STATUS_PLAYING,
        "fen": ChessConstants.STARTING_FEN,
        "history_fens": [],
        "ai_state": {
            "board_state": {},
            "new_step": "",
            "new_fen": ""
        }
    }

def referee_node(state: GameState) -> Dict[str, Any]:
    """
    Check game status and determine next action.
    Evaluates for checkmate, draw, resignation or continues play.
    """

    if state["fen"] == "----":
        return {"status": ChessConstants.STATUS_RESIGNATION, "history_fens": [state["fen"]]}

    try:
        board = chess.Board(state["fen"])

        # Check game status
        if board.is_checkmate():
            return {"status": ChessConstants.STATUS_MATE, "history_fens": [state["fen"]]}
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return {"status": ChessConstants.STATUS_DRAW, "history_fens": [state["fen"]]}
        else:
            return {"status": ChessConstants.STATUS_PLAYING, "history_fens": [state["fen"]]}
    except ValueError:
        # Invalid FEN, restore from history
        if state["history_fens"]:
            return {"fen": state["history_fens"][-1], "status": ChessConstants.STATUS_PLAYING}
        return {"status": ChessConstants.STATUS_DRAW}

def should_play_or_finish(state: GameState) -> str:
    """Conditional edge function to determine if game should end."""
    if state["status"] in [ChessConstants.STATUS_MATE, ChessConstants.STATUS_DRAW, ChessConstants.STATUS_RESIGNATION]:
        return END
    elif state["turn"] == ChessConstants.TURN_USER:
        return GraphNode.USER_STEP
    else:
        return GraphNode.AI_STEP

def user_step_node(state: GameState) -> GameState:
    """
    Handle user move input and validation.
    The user's FEN input comes from Command(resume=...) and is available in state.
    """
    # Get the user's FEN from the resumed state
    new_fen = interrupt("fetch new fen")

    # Validate the received FEN
    if not is_valid_fen(new_fen):
        # Restore from history if invalid
        if state["history_fens"]:
            return {"fen": state["history_fens"][-1]}
        return state
    
    # Check if transition is legal
    if len(state["history_fens"]) > 0:
        prev_fen = state["history_fens"][-1]
        if is_legal_fen_transition(prev_fen, new_fen):
            return {
                "fen": new_fen,
                "turn": ChessConstants.TURN_AI
            }
    
    # If we reach here, move was illegal - restore previous FEN
    if state["history_fens"]:
        return {"fen": state["history_fens"][-1]}

    return state

# AI Subgraph Nodes

def get_board_state(state: GameState) -> GameState:
    """
    AI agent prepare board state from board fen.
    """
    ai_state = state["ai_state"].copy()
    ai_state["board_state"] = fen_to_json(state["fen"])
    return { "ai_state": ai_state }

def defend_node(state: GameState) -> Command[Literal[GraphNode.ATTACK_STEP, GraphNode.MAKE_FEN]]:
    """
    AI agent checks if he needs to defend from opponent.
    If true then do appropriate step and ignore other step nodes.
    """
    ai_state = state["ai_state"].copy()
    new_step = "----"

    try:
        # Create system prompt
        prompt = read_file(FilePathConstants.PROMPT_AGENT_DEFEND)
        system_prompt = prompt.format(
            ai_color=state["ai_color"],
            opponent_color=state["user_color"],
            chess_rules=chess_rules,
            board_description=ai_state["board_state"],
            board_fen=state["fen"]
        )

        tools = [check_step_is_legal, get_new_board_state]

        agent_executor = create_agent(tools, system_prompt)
        config = create_agent_config(GraphNode.DEFEND_STEP)

        # Execute agent with retry
        response = agent_executor.invoke({"messages": [{"role": "user", "content": human_prompt}]}, config=config)
        new_step = get_step_from_last_response_message(response)

    except Exception as e:
        print(f"Error in move_node: {e}")

    ai_state["new_step"] = new_step
    return Command(
        update={"ai_state": ai_state},
        goto=GraphNode.MAKE_FEN if new_step != "----" else GraphNode.ATTACK_STEP
    )

def attack_node(state: GameState) -> Command[Literal[GraphNode.PREDICTED_STEP, GraphNode.MAKE_FEN]]:
    """
    AI agent checks if he needs to attack opponent.
    If true then do appropriate step and ignore other step nodes.
    """
    ai_state = state["ai_state"].copy()
    new_step = "----"

    try:
        # Create system prompt
        prompt = read_file(FilePathConstants.PROMPT_AGENT_ATTACK)
        system_prompt = prompt.format(
            ai_color=state["ai_color"],
            opponent_color=state["user_color"],
            chess_rules=chess_rules,
            board_description=ai_state["board_state"],
            board_fen=state["fen"]
        )

        tools = [check_step_is_legal, get_new_board_state]

        agent_executor = create_agent(tools, system_prompt)
        config = create_agent_config(GraphNode.ATTACK_STEP)

        # Execute agent with retry
        response = agent_executor.invoke({"messages": [{"role": "user", "content": human_prompt}]}, config=config)
        new_step = get_step_from_last_response_message(response)

    except Exception as e:
        print(f"Error in move_node: {e}")

    ai_state["new_step"] = new_step
    return Command(
        update={"ai_state": ai_state},
        goto=GraphNode.MAKE_FEN if new_step != "----" else GraphNode.PREDICTED_STEP
    )

def predicted_steps_node(state: GameState) -> Command[Literal[GraphNode.STRATEGY_STEP, GraphNode.MAKE_FEN]]:
    """
    AI agent tries to do step based on history-steps of strong chess-players with similar board.
    if step done then ignore other step nodes.
    """
    ai_state = state["ai_state"].copy()
    new_step = "----"

    try:
        # Format possible steps
        possible_steps = get_possible_steps(fen=state["fen"], ai_color=state["ai_color"])

        # Create system prompt
        prompt = read_file(FilePathConstants.PROMPT_AGENT_PREDICTED_STEPS)
        system_prompt = prompt.format(
            ai_color=state["ai_color"],
            opponent_color=state["user_color"],
            chess_rules=chess_rules,
            board_description=ai_state["board_state"],
            board_fen=state["fen"],
            possible_steps=possible_steps
        )

        tools = [check_step_is_legal]

        agent_executor = create_agent(tools, system_prompt)
        config = create_agent_config(GraphNode.PREDICTED_STEP)

        # Execute agent with retry
        response = agent_executor.invoke({"messages": [{"role": "user", "content": human_prompt}]}, config=config)
        new_step = get_step_from_last_response_message(response)

    except Exception as e:
        print(f"Error in move_node: {e}")

    ai_state["new_step"] = new_step
    return Command(
        update={"ai_state": ai_state},
        goto=GraphNode.MAKE_FEN if new_step != "----" else GraphNode.STRATEGY_STEP
    )

def strategy_step_node(state: GameState) -> Command[Literal[GraphNode.MAKE_FEN]]:
    """
    AI agent tries to do step based on tactic and strategy instructions.
    """
    ai_state = state["ai_state"].copy()
    new_step = "----"

    try:
        # Create system prompt
        prompt = read_file(FilePathConstants.PROMPT_AGENT_STRATEGY_STEP)
        system_prompt = prompt.format(
            ai_color=state["ai_color"],
            opponent_color=state["user_color"],
            chess_rules=chess_rules,
            board_description=ai_state["board_state"],
            board_fen=state["fen"]
        )

        tools = [check_step_is_legal]

        agent_executor = create_agent(tools, system_prompt)
        config = create_agent_config(GraphNode.STRATEGY_STEP)

        # Execute agent with retry
        response = agent_executor.invoke({"messages": [{"role": "user", "content": human_prompt}]}, config=config)
        new_step = get_step_from_last_response_message(response)

    except Exception as e:
        print(f"Error in move_node: {e}")

    ai_state["new_step"] = new_step
    return Command(
        update={"ai_state": ai_state},
        goto=GraphNode.MAKE_FEN
    )

def make_fen_node(state: GameState) -> GameState:
    """
    Validate AI's proposed move and create new board fen.
    """
    ai_state = state["ai_state"].copy()
    current_fen = state["fen"]
    new_step = state["ai_state"]["new_step"]

    if new_step != "----" and is_legal_step(fen=current_fen, uci=new_step):
        new_fen = apply_uci_move(fen=current_fen, uci=new_step)
        ai_state["new_fen"] = new_fen
    else:
        ai_state["new_fen"] = "----"

    return {"ai_state": ai_state}

# Create graphs
def create_ai_subgraph():
    """Create the AI move subgraph."""
    ai_graph = StateGraph(GameState)

    ai_graph.add_node(GraphNode.GET_BOARD, get_board_state)
    ai_graph.add_node(GraphNode.DEFEND_STEP, defend_node)
    ai_graph.add_node(GraphNode.ATTACK_STEP, attack_node)
    ai_graph.add_node(GraphNode.PREDICTED_STEP, predicted_steps_node)
    ai_graph.add_node(GraphNode.STRATEGY_STEP, strategy_step_node)
    ai_graph.add_node(GraphNode.MAKE_FEN, make_fen_node)

    ai_graph.add_edge(START, GraphNode.GET_BOARD)
    ai_graph.add_edge(GraphNode.GET_BOARD, GraphNode.DEFEND_STEP)
    ai_graph.add_edge(GraphNode.MAKE_FEN, END)

    compiled_graph = ai_graph.compile(checkpointer=checkpointer)

    post_process = RunnableLambda(lambda state: {
        "fen":  state["ai_state"]["new_fen"],
        "turn": ChessConstants.TURN_USER if state["ai_state"]["new_fen"] != "----" else ChessConstants.TURN_AI,
        "ai_state": {
            "board_state": {},
            "new_step": "",
            "new_fen": ""
        }
    })
    graph_node = compiled_graph | post_process
    return graph_node

def create_chess_graph():
    """
    Create the main chess game graph.
    Returns compiled graph ready for execution.
    """
    # Init globals
    init_globals()

    # Create main graph
    graph = StateGraph(GameState)
    
    # Add nodes
    graph.add_node(GraphNode.START, start_node)
    graph.add_node(GraphNode.REFEREE, referee_node)
    graph.add_node(GraphNode.USER_STEP, user_step_node)
    
    # Create AI subgraph
    ai_subgraph = create_ai_subgraph()
    graph.add_node(GraphNode.AI_STEP, ai_subgraph)
    
    # Add edges
    graph.add_edge(START, GraphNode.START)
    graph.add_edge(GraphNode.START, GraphNode.REFEREE)
    graph.add_edge(GraphNode.USER_STEP, GraphNode.REFEREE)
    graph.add_edge(GraphNode.AI_STEP, GraphNode.REFEREE)
    
    # Add conditional edges from referee
    graph.add_conditional_edges(
        GraphNode.REFEREE,
        should_play_or_finish,
        {
            GraphNode.USER_STEP: GraphNode.USER_STEP,
            GraphNode.AI_STEP: GraphNode.AI_STEP,
            END: END
        }
    )

    # Compile with interrupt for user input
    return graph.compile(checkpointer=checkpointer)

# Example usage
if __name__ == "__main__":
    chess_graph = create_chess_graph()
    
    # Test initial state
    initial_state = {
        "user_color": ChessConstants.COLOR_WHITE,
        "ai_color": ChessConstants.COLOR_BLACK
    }

    result = chess_graph.invoke(initial_state, config=ConfigConstants.CHESS_GAME_CONFIG)
    print("Initial game state:", result)

    resume_value = "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"
    result = chess_graph.invoke(Command(resume=resume_value), config=ConfigConstants.CHESS_GAME_CONFIG)
    print("New game state:", result)