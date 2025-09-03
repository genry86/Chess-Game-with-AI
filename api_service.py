"""
API service for chess game backend.
Provides REST endpoints for SwiftUI macOS app to interact with chess game logic.
Uses Uvicorn server and manages chess graph state globally.
"""
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import uvicorn
from langgraph.types import Command

from utility import load_rag
from game_graph import create_chess_graph
from constants import ConfigConstants, APIConstants, ChessConstants

# CORS middleware for local development
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# Global variables to store graph and state
chess_graph = None
graph_state = None

class StartGameRequest(BaseModel):
    """Request model for starting a new game."""
    user_color: str  # 'white' or 'black'


class PlayMoveRequest(BaseModel):
    """Request model for playing a move."""
    fen: str


class GameResponse(BaseModel):
    """Response model for game state."""
    fen: str
    status: str
    turn: str
    message: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    print("Chess Game API starting up...")
    print("Available endpoints:")
    print("  POST /start - Start a new game")
    print("  POST /play - Make a move")
    print("  GET /status - Get game status")
    print("  GET /health - Health check")
    load_rag()

    yield
    
    # Shutdown
    print("Chess Game API shutting down...")


# Initialize FastAPI app
app = FastAPI(title="Chess Game API", version="1.0.0", lifespan=lifespan)


@app.post(APIConstants.ENDPOINT_START, response_model=GameResponse)
async def start_game(request: StartGameRequest):
    """
    Start or restart a chess game.
    
    Args:
        request: Contains user's color choice ('white' or 'black')
        
    Returns:
        GameResponse with initial game state
    """
    global chess_graph, graph_state

    try:
        # Validate user color choice
        if request.user_color not in [ChessConstants.COLOR_WHITE, ChessConstants.COLOR_BLACK]:
            raise HTTPException(status_code=400, detail=f"User color must be '{ChessConstants.COLOR_WHITE}' or '{ChessConstants.COLOR_BLACK}'")
        
        # Determine AI color
        ai_color = ChessConstants.COLOR_BLACK if request.user_color == ChessConstants.COLOR_WHITE else ChessConstants.COLOR_WHITE

        # Create new chess graph
        chess_graph = create_chess_graph()
        
        # Initialize game state
        initial_state = {
            "user_color": request.user_color,
            "ai_color": ai_color
        }
        
        # Invoke the graph to initialize
        graph_state = chess_graph.invoke(initial_state, ConfigConstants.CHESS_GAME_CONFIG)

        return GameResponse(
            fen=graph_state["fen"],
            status=graph_state["status"],
            turn=graph_state["turn"],
            message=f"Game started. You are playing as {request.user_color}."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start game: {str(e)}")


@app.post(APIConstants.ENDPOINT_PLAY, response_model=GameResponse)
async def play_move(request: PlayMoveRequest):
    """
    Process a user move and get AI response.

    Args:
        request: Contains the FEN string after user's move
        
    Returns:
        GameResponse with updated game state after AI move
    """
    global chess_graph, graph_state
    
    try:
        # Check if game is initialized
        if chess_graph is None or graph_state is None:
            raise HTTPException(status_code=400, detail="Game not started. Call /start first.")

        # Validate FEN format (basic check)
        if not request.fen or len(request.fen.strip()) == 0:
            raise HTTPException(status_code=400, detail="Invalid FEN string provided")

        resume_value = request.fen.strip()
        graph_state = chess_graph.invoke(Command(resume=resume_value), ConfigConstants.CHESS_GAME_CONFIG)


        # Determine response message based on game status
        message = None
        if graph_state["status"] == ChessConstants.STATUS_MATE:
            if graph_state["turn"] == ChessConstants.TURN_USER:
                message = "Checkmate! AI wins."
            else:
                message = "Checkmate! You win!"
        elif graph_state["status"] == ChessConstants.STATUS_DRAW:
            message = "Game ended in a draw."
        elif graph_state["turn"] == ChessConstants.TURN_USER:
            message = "Your turn."
        else:
            message = "AI is thinking..."

        return GameResponse(
            fen=graph_state["fen"],
            status=graph_state["status"],
            turn=graph_state["turn"],
            message=message
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process move: {str(e)}")


@app.get(APIConstants.ENDPOINT_STATUS, response_model=GameResponse)
async def get_game_status():
    """
    Get current game status without making a move.
    
    Returns:
        GameResponse with current game state
    """
    global graph_state
    
    try:
        if graph_state is None:
            raise HTTPException(status_code=400, detail="No active game. Call /start first.")
        
        return GameResponse(
            fen=graph_state["fen"],
            status=graph_state["status"],
            turn=graph_state["turn"],
            message="Current game status"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.get(APIConstants.ENDPOINT_HEALTH)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Chess API is running"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_server():
    """
    Run the Uvicorn server.
    Default configuration for local development.
    """
    uvicorn.run(
        "api_service:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":

    print("Starting Chess Game API server...")
    print("Server will run on http://127.0.0.1:8000")
    print("API docs available at http://127.0.0.1:8000/docs")
    run_server()