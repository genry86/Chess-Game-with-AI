//
//  APIManager.swift
//  chess
//
//  Manages communication with the chess game API
//

import Foundation
import Combine

/// Manager for handling API communication with the chess backend
final class APIManager: ObservableObject {
    
    @Published private(set) var isLoading = false
    @Published private(set) var lastError: APIError?
    
    private let session: URLSession
    private let baseURL: String
    
    init(baseURL: String = Constants.API.baseURL) {
        self.baseURL = baseURL
        
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = Constants.API.timeout
        config.timeoutIntervalForResource = Constants.API.timeout
        self.session = URLSession(configuration: config)
    }
}

// MARK: - Public API Methods
extension APIManager {
    
    /// Start a new chess game with specified user color
    /// - Parameter userColor: The color the user wants to play as
    /// - Returns: Initial game state
    func startGame(userColor: ChessColor) async throws -> GameState {
        isLoading = true
        lastError = nil
        
        defer { isLoading = false }
        
        let request = StartGameRequest(userColor: userColor.rawValue)
        let url = URL(string: baseURL + Constants.API.startEndpoint)!
        
        return try await performRequest(url: url, method: "POST", body: request)
    }
    
    /// Send a move to the server and get updated game state
    /// - Parameter fen: The FEN string after user's move
    /// - Returns: Updated game state after AI response
    func playMove(fen: String) async throws -> GameState {
        isLoading = true
        lastError = nil
        
        defer { isLoading = false }
        
        let request = PlayMoveRequest(fen: fen)
        let url = URL(string: baseURL + Constants.API.playEndpoint)!
        
        return try await performRequest(url: url, method: "POST", body: request)
    }
    
    /// Get current game status without making a move
    /// - Returns: Current game state
    func getGameStatus() async throws -> GameState {
        isLoading = true
        lastError = nil
        
        defer { isLoading = false }
        
        let url = URL(string: baseURL + Constants.API.statusEndpoint)!
        
        return try await performRequest(url: url, method: "GET", body: nil as Empty?)
    }
    
    /// Check if the API server is healthy
    /// - Returns: True if server is responding
    func healthCheck() async -> Bool {
        do {
            let url = URL(string: baseURL + Constants.API.healthEndpoint)!
            let (_, response) = try await session.data(from: url)
            
            if let httpResponse = response as? HTTPURLResponse {
                return httpResponse.statusCode == 200
            }
            return false
        } catch {
            return false
        }
    }
}

// MARK: - Private Helper Methods
private extension APIManager {
    
    /// Perform HTTP request with optional body
    /// - Parameters:
    ///   - url: Request URL
    ///   - method: HTTP method
    ///   - body: Optional request body (for POST requests)
    /// - Returns: Decoded response
    func performRequest<T: Codable, U: Codable>(
        url: URL,
        method: String,
        body: T? = nil
    ) async throws -> U {
        
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        
        // Add request body if provided
        if let body = body {
            let encoder = JSONEncoder()
            request.httpBody = try encoder.encode(body)
        }
        
        do {
            let (data, response) = try await session.data(for: request)
            
            // Check HTTP status code
            if let httpResponse = response as? HTTPURLResponse {
                if httpResponse.statusCode >= 400 {
                    // Try to decode error response
                    let decoder = JSONDecoder()
                    if let apiError = try? decoder.decode(APIError.self, from: data) {
                        lastError = apiError
                        throw apiError
                    } else {
                        let error = APIError(detail: "HTTP \(httpResponse.statusCode): Request failed")
                        lastError = error
                        throw error
                    }
                }
            }
            
            // Decode successful response
            let decoder = JSONDecoder()
            return try decoder.decode(U.self, from: data)
            
        } catch let error as APIError {
            lastError = error
            throw error
        } catch {
            let apiError = APIError(detail: "Network error: \(error.localizedDescription)")
            lastError = apiError
            throw apiError
        }
    }
}

// MARK: - Error Handling
extension APIManager {
    
    /// Clear the last error
    func clearError() {
        lastError = nil
    }
    
    /// Check if there's an active error
    var hasError: Bool {
        return lastError != nil
    }
}

// MARK: - Empty Request Model
private struct Empty: Codable {}
