//
//  ColorSelectionView.swift
//  chess
//
//  Modal view for selecting chess piece color
//

import SwiftUI

struct ColorSelectionView: View {
    let onColorSelected: (ChessColor) -> Void
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        VStack(spacing: Constants.UI.standardPadding * 2) {
            // Title
            Text("Choose Your Color")
                .font(.title)
                .fontWeight(.bold)
                .multilineTextAlignment(.center)
            
            Text("Select which color you'd like to play as")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            
            // Color selection buttons
            HStack(spacing: Constants.UI.standardPadding * 2) {
                ColorOptionButton(
                    color: .white,
                    title: "White",
                    subtitle: "You move first",
                    systemImage: "circle"
                ) {
                    selectColor(.white)
                }
                
                ColorOptionButton(
                    color: .black,
                    title: "Black",
                    subtitle: "AI moves first",
                    systemImage: "circle.fill"
                ) {
                    selectColor(.black)
                }
            }
            
            // Cancel button
            Button("Cancel") {
                dismiss()
            }
            .foregroundStyle(.secondary)
        }
        .padding(Constants.UI.standardPadding * 2)
        .frame(width: 400)
    }
}

// MARK: - Private Methods
private extension ColorSelectionView {
    
    /// Handle color selection
    /// - Parameter color: Selected chess color
    func selectColor(_ color: ChessColor) {
        onColorSelected(color)
        dismiss()
    }
}

// MARK: - Color Option Button
private struct ColorOptionButton: View {
    let color: ChessColor
    let title: String
    let subtitle: String
    let systemImage: String
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 12) {
                Image(systemName: systemImage)
                    .font(.system(size: 40))
                    .foregroundStyle(color == .white ? .primary : .primary)
                
                VStack(spacing: 4) {
                    Text(title)
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(width: 140, height: 120)
            .background(Color(.controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: Constants.UI.cornerRadius))
            .overlay(
                RoundedRectangle(cornerRadius: Constants.UI.cornerRadius)
                    .stroke(Color(.separatorColor), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .scaleEffect(1.0)
        .animation(.easeInOut(duration: Constants.Animation.quick), value: false)
    }
}
