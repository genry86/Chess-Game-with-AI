"""
Chess Application Constants
Minimal constants file containing only the most essential values.
"""

class APIConstants:
    """Constants for API endpoints."""
    ENDPOINT_START = "/start"
    ENDPOINT_PLAY = "/play"
    ENDPOINT_STATUS = "/status"
    ENDPOINT_HEALTH = "/health"


class ChessConstants:
    """Constants for chess game logic."""
    # Chess Colors
    COLOR_WHITE = "white"
    COLOR_BLACK = "black"
    
    # Game Status Values
    STATUS_PLAYING = "playing"
    STATUS_MATE = "mate"
    STATUS_DRAW = "draw"
    STATUS_RESIGNATION = "resignation"

    # Turn Indicators
    TURN_USER = "user"
    TURN_AI = "ai"
    
    # Initial Chess Position
    STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # STARTING_FEN = "r1bqkb1r/pppp1ppp/2npp3/4p3/2B1P1b1/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5"   # black bishop attacks white knight
    # STARTING_FEN = "rnbqkb1r/pppp1ppp/3p4/4pp1b/2B1P3/Q5N1/PPPP1PPP/R1B1K1NR w KQkq - 4 5"   # white knight attacks black bishop
    # STARTING_FEN = "rnb1kb1r/pppp1p1p/3p2p1/4pq1b/2B5/Q3P1N1/PPPP1PPP/R1B1K1NR w KQkq - 4 5"   # white knight attacks black bishop and queen
    # STARTING_FEN = "rn1qkb1r/ppp2pp1/3pb2p/4p3/2B1P3/Q5N1/PPPP1PPP/R1B1K1NR w KQkq - 4 5"   # white bishop vs black bishop


class AIConstants:
    """Constants for AI configuration."""
    # OpenAI Model Configuration
    TEMPERATURE = 0.0
    CONTEXT_WINDOW = 32000
    RECURSION_LIMIT = 100
    
    # RAG System Configuration
    RAG_RESULTS_COUNT = 100
    
    # Retry Configuration
    MAX_RETRY_ATTEMPTS = 3


class FilePathConstants:
    """Constants for file and directory paths."""
    # Prompt File Paths
    PROMPT_AGENT_DEFEND = "./prompts/agent_defend_prompt.txt"
    PROMPT_AGENT_ATTACK = "./prompts/agent_attack_prompt.txt"
    PROMPT_AGENT_PREDICTED_STEPS = "./prompts/agent_predicted_steps_prompt.txt"
    PROMPT_AGENT_STRATEGY_STEP = "./prompts/agent_strategy_step_prompt.txt"
    PROMPT_AGENT_START = "./prompts/agent_user_prompt.txt"

    PROMPT_CHESS_RULES = "./prompts/chess_rules_description.txt"
    # PROMPT_FEW_SHOT_EXAMPLES = "./prompts/few_show_examples.txt"
    
    # Database Paths
    DEFAULT_DATABASE_PATH = "./RAG/database/"
    DATABASE_SUBDIRECTORY = "chroma"
    
    # Data Directories
    DATA_DIRECTORY = "./data"
    DATABASE_DIRECTORY = "./database"
    CHECKPOINTER_DATABASE = "./checkpointer/chess_checkpoints.db"
    DB_URI = "postgresql://user:pass@host:5432/dbname"


class DatabaseConstants:
    """Constants for database configuration."""
    DEFAULT_COLLECTION_NAME = "chess_games"


class ConfigConstants:
    """Constants for configuration objects."""
    # Chess game thread configuration
    CHESS_GAME_CONFIG = {
        "configurable": {"thread_id": "chess_game_thread"},
        "recursion_limit": AIConstants.RECURSION_LIMIT
    }


class GraphNode:
    """Constants for graph node names."""
    # Main Graph Nodes
    START = "start"
    REFEREE = "referee"
    USER_STEP = "user_step"
    AI_STEP = "ai_step"
    
    # AI Subgraph Nodes
    GET_BOARD = "get_board"
    DEFEND_STEP = "defend_step"
    ATTACK_STEP = "attack_step"
    PREDICTED_STEP = "predicted_step"
    STRATEGY_STEP = "strategy_step"
    MAKE_FEN = "make_fen"