//
//  ChessGameViewModel.swift
//  chess
//
//  View model for chess game logic and state management
//

import Foundation
import Combine
import ChessboardKit

/// View model managing chess game state and API communication
@MainActor
final class ChessGameViewModel: ObservableObject {
    
    @Published private(set) var gameState: GameState?
    @Published private(set) var chessboardModel: ChessboardModel
    @Published private(set) var isLoading = false
    @Published private(set) var lastError: APIError?
    
    private let apiManager: APIManager
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        self.apiManager = APIManager()
        self.chessboardModel = ChessboardModel(fen: Constants.Chess.initialFEN)
        
        setupBindings()
    }
}

// MARK: - Public Methods
extension ChessGameViewModel {
    
    /// Start a new chess game with the specified user color
    /// - Parameter userColor: The color the user wants to play as
    func startNewGame(userColor: ChessColor) async {
        do {
            let newGameState = try await apiManager.startGame(userColor: userColor)
            await updateGameState(newGameState)
        } catch {
            if let apiError = error as? APIError {
                lastError = apiError
            } else {
                lastError = APIError(detail: error.localizedDescription)
            }
        }
    }
    
    /// Handle user move on the chessboard
    /// - Parameter newFEN: The FEN after user's move
    func handleUserMove(newFEN: String) async {
        guard let currentState = gameState,
              currentState.isGameActive,
              currentState.isUserTurn else {
            return
        }
        
        do {
            let updatedState = try await apiManager.playMove(fen: newFEN)
            await updateGameState(updatedState)
        } catch {
            if let apiError = error as? APIError {
                lastError = apiError
            } else {
                lastError = APIError(detail: error.localizedDescription)
            }
            
            // Revert to previous position on error
            if let currentFEN = gameState?.fen {
                updateChessboard(fen: currentFEN)
            }
        }
    }
    
    /// Clear the current error
    func clearError() {
        lastError = nil
    }
    
    /// Check if there's an active error
    var hasError: Bool {
        return lastError != nil
    }
}

// MARK: - Private Methods
private extension ChessGameViewModel {
    
    /// Set up reactive bindings for API manager
    func setupBindings() {
        // Observe API manager loading state
        apiManager.$isLoading
            .receive(on: DispatchQueue.main)
            .assign(to: \.isLoading, on: self)
            .store(in: &cancellables)
        
        // Observe API manager errors
        apiManager.$lastError
            .receive(on: DispatchQueue.main)
            .assign(to: \.lastError, on: self)
            .store(in: &cancellables)
    }
    
    /// Update game state and chessboard
    /// - Parameter newState: New game state from API
    func updateGameState(_ newState: GameState) async {
        gameState = newState
        updateChessboard(fen: newState.fen)
    }
    
    /// Update chessboard with new FEN
    /// - Parameter fen: New FEN string
    func updateChessboard(fen: String) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.chessboardModel.setFen(fen)
        }
    }
}
