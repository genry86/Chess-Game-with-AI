//
//  ChessGameView.swift
//  chess
//
//  Main chess game view with chessboard and game controls
//

import SwiftUI
import ChessboardKit

struct ChessGameView: View {
    @StateObject private var viewModel = ChessGameViewModel()
    @State private var showingColorSelection = false
    
    var body: some View {
        VStack(spacing: Constants.UI.standardPadding) {
            // Game status message
            if let message = viewModel.gameState?.message {
                Text(message)
                    .font(.headline)
                    .foregroundStyle(.primary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }
            
            // Chessboard
            ChessboardContainerView(viewModel: viewModel)
            
            // Start new game button
            Button("Start New Game") {
                showingColorSelection = true
            }
            .frame(height: Constants.UI.buttonHeight)
            .frame(maxWidth: .infinity)
            .background(Color.accentColor)
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: Constants.UI.cornerRadius))
            .padding(.horizontal)
            .disabled(viewModel.isLoading)
            
            Spacer()
        }
        .frame(width: Constants.UI.windowWidth, height: Constants.UI.windowHeight)
        .padding()
        .sheet(isPresented: $showingColorSelection) {
            ColorSelectionView { selectedColor in
                Task {
                    await viewModel.startNewGame(userColor: selectedColor)
                }
            }
        }
        .alert("Error", isPresented: .constant(viewModel.hasError)) {
            Button("OK") {
                viewModel.clearError()
            }
        } message: {
            if let error = viewModel.lastError {
                Text(error.localizedDescription)
            }
        }
        .overlay {
            if viewModel.isLoading {
                LoadingOverlay()
            }
        }
        .onAppear {
            showingColorSelection = true
        }
    }
}
