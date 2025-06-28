import SwiftUI
import PhotosUI

// MARK: - VideoItem Transferable
struct VideoItem: Transferable {
    let url: URL
    
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            // Copy to a temporary location to ensure we have access
            let tempDir = FileManager.default.temporaryDirectory
            let fileName = "\(UUID().uuidString).\(received.file.pathExtension)" // Ensure unique filename
            let copyURL = tempDir.appendingPathComponent(fileName)
            
            // Attempt to remove existing file at destination URL before copying
            try? FileManager.default.removeItem(at: copyURL)

            try FileManager.default.copyItem(at: received.file, to: copyURL)
            return Self.init(url: copyURL)
        }
    }
}
