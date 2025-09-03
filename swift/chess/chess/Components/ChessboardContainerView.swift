//
//  ChessboardContainerView.swift
//  chess
//
//  Container view for the chessboard with move handling
//

import SwiftUI
import ChessboardKit

struct ChessboardContainerView: View {
    @ObservedObject var viewModel: ChessGameViewModel
    
    var body: some View {
        Chessboard(chessboardModel: viewModel.chessboardModel)
            .onMove { move, isLegal, from, to, lan, promotionPiece in
                handleMove(
                    move: move,
                    isLegal: isLegal,
                    from: from,
                    to: to,
                    lan: lan,
                    promotionPiece: promotionPiece
                )
            }
            .frame(
                width: Constants.UI.chessboardSize,
                height: Constants.UI.chessboardSize
            )
            .disabled(viewModel.isLoading || !(viewModel.gameState?.isUserTurn ?? false))
    }
}

// MARK: - Private Methods
private extension ChessboardContainerView {
    
    /// Handle chess move on the board
    /// - Parameters:
    ///   - move: The chess move object
    ///   - isLegal: Whether the move is legal
    ///   - from: Source square
    ///   - to: Target square
    ///   - lan: Long algebraic notation
    ///   - promotionPiece: Promotion piece if applicable
    func handleMove(
        move: Any,
        isLegal: Bool,
        from: String,
        to: String,
        lan: String,
        promotionPiece: Any?
    ) {
        print("Move attempted: \(lan), isLegal: \(isLegal), from: \(from), to: \(to)")
        
        guard isLegal else {
            print("Illegal move attempted: \(lan)")
            return
        }
        
        // Make the move on the chessboard model using LAN notation
        viewModel.chessboardModel.game.make(move: lan)
        
        // Get the new FEN after the move
        let newFEN = viewModel.chessboardModel.fen
        print("New FEN after user move: \(newFEN)")
        
        // Update the chessboard with the new position
        viewModel.chessboardModel.setFen(newFEN, lan: lan)
        
        // Send the move to the server
        Task {
            await viewModel.handleUserMove(newFEN: newFEN)
        }
    }
}
