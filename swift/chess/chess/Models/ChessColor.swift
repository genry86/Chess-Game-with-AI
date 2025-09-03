//
//  ChessColor.swift
//  chess
//
//  Chess color enumeration for player choices
//

import Foundation

/// Represents chess piece colors
enum ChessColor: String, CaseIterable {
    case white = "white"
    case black = "black"
    
    /// Opposite color
    var opposite: ChessColor {
        switch self {
        case .white:
            return .black
        case .black:
            return .white
        }
    }
    
    /// Display name for UI
    var displayName: String {
        switch self {
        case .white:
            return "White"
        case .black:
            return "Black"
        }
    }
}
