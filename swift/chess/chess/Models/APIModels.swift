//
//  APIModels.swift
//  chess
//
//  API request and response models for chess game communication
//

import Foundation

/// Request model for starting a new chess game
struct StartGameRequest: Codable {
    let userColor: String
    
    enum CodingKeys: String, CodingKey {
        case userColor = "user_color"
    }
}

/// Request model for making a chess move
struct PlayMoveRequest: Codable {
    let fen: String
}

/// Response model for API errors
struct APIError: Codable, Error {
    let detail: String
    
    /// Human-readable error description
    var localizedDescription: String {
        return detail
    }
}
