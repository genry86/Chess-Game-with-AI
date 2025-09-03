//
//  Constants.swift
//  chess
//
//  Application constants and configuration
//

import Foundation

/// Application-wide constants
struct Constants {
    
    /// API configuration
    struct API {
        static let baseURL = "http://127.0.0.1:8000"
        static let startEndpoint = "/start"
        static let playEndpoint = "/play"
        static let statusEndpoint = "/status"
        static let healthEndpoint = "/health"
        static let timeout: TimeInterval = 900.0
    }
    
    /// UI configuration
    struct UI {
        static let windowWidth: CGFloat = 500
        static let windowHeight: CGFloat = 500
        static let chessboardSize: CGFloat = 400
        static let buttonHeight: CGFloat = 44
        static let cornerRadius: CGFloat = 8
        static let standardPadding: CGFloat = 16
    }
    
    /// Chess game configuration
    struct Chess {
        static let initialFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    }
    
    /// Animation durations
    struct Animation {
        static let standard: Double = 0.3
        static let quick: Double = 0.15
    }
}
