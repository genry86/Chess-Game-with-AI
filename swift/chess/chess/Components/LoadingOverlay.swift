//
//  LoadingOverlay.swift
//  chess
//
//  Loading overlay component with activity indicator
//

import SwiftUI

struct LoadingOverlay: View {
    var body: some View {
        ZStack {
            // Semi-transparent background
            Color.black
                .opacity(0.3)
                .ignoresSafeArea()
            
            // Loading content
            VStack(spacing: Constants.UI.standardPadding) {
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle())
                    .scaleEffect(1.2)
                
                Text("Thinking...")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .padding(Constants.UI.standardPadding * 2)
            .background(.regularMaterial)
            .clipShape(RoundedRectangle(cornerRadius: Constants.UI.cornerRadius))
        }
        .animation(.easeInOut(duration: Constants.Animation.standard), value: true)
    }
}
