//
//  ChessApp.swift
//  chess
//
//  Main chess application entry point
//

import SwiftUI

@main
struct ChessApp: App {
    var body: some Scene {
        WindowGroup {
            ChessGameView()
        }
        .windowResizability(.contentSize)
    }
}
