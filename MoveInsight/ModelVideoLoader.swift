import Foundation
import AVKit

// This class helps manage model videos in the app's bundle
class ModelVideoLoader {
    static let shared = ModelVideoLoader()
    
    // Map of technique names to their video filenames (without extension)
    private let techniqueVideoMap: [String: String] = [
        "Underhand Clear": "underhand_clear_model",
        // Other techniques would be added here as they become available
    ]
    
    private init() {
        // Log available model videos on initialization
        print("ModelVideoLoader initialized with techniques: \(techniqueVideoMap.keys.joined(separator: ", "))")
    }
    
    // Get the URL for a specific technique's model video
    func getModelVideoURL(for technique: String) -> URL? {
        // Try to find the exact technique name in our map
        let cleanTechniqueName = technique.trimmingCharacters(in: .whitespacesAndNewlines)
        
        if let filename = techniqueVideoMap[cleanTechniqueName] {
            if let url = Bundle.main.url(forResource: filename, withExtension: "mov") {
                print("Found model video for \(cleanTechniqueName) at \(url.path)")
                return url
            }
        }
        
        // Try with a normalized technique name (lowercase, underscores)
        let normalizedName = cleanTechniqueName.lowercased().replacingOccurrences(of: " ", with: "_")
        if let url = Bundle.main.url(forResource: normalizedName, withExtension: "mov") {
            print("Found model video using normalized name \(normalizedName)")
            return url
        }
        
        // Special case for backhand clear (our only implemented video currently)
        if cleanTechniqueName.lowercased().contains("backhand") {
            if let fallbackURL = Bundle.main.url(forResource: "backhand_clear_model", withExtension: "mov") {
                print("Found underhand clear model video as fallback")
                return fallbackURL
            }
        }
        
        print("No model video found for technique: \(technique)")
        return nil
    }
    
    // Check if a model video exists for a specific technique
    func hasModelVideo(for technique: String) -> Bool {
        return getModelVideoURL(for: technique) != nil
    }
    
    // Create a VideoPlayerViewModel for a model video
    func createModelVideoViewModel(for technique: String) -> VideoPlayerViewModel? {
        guard let videoURL = getModelVideoURL(for: technique) else {
            print("Failed to create model ViewModel: no URL found for \(technique)")
            return nil
        }
        
        print("Creating model VideoPlayerViewModel for \(technique)")
        return VideoPlayerViewModel(
            videoURL: videoURL,
            videoSource: .secondary
        )
    }
}
