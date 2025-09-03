//
//  GameState.swift
//  chess
//
//  Chess game state model for API communication
//

import Foundation

/// Represents the current state of a chess game
struct GameState: Codable {
    let fen: String
    let status: String  // "playing", "mate", "draw"
    let turn: String    // "user", "ai"
    let message: String?
    
    /// Check if the game is active (not ended)
    var isGameActive: Bool {
        return status == "playing"
    }
    
    /// Check if it's user's turn
    var isUserTurn: Bool {
        return turn == "user"
    }
    
    /// Check if game has ended
    var isGameEnded: Bool {
        return status == "mate" || status == "draw"
    }
}
