// MoveInsight/TechniqueAnalysisService.swift
import Foundation
import Combine
import AVFoundation // For URL

// MARK: - Data Structures for Server Communication

struct ServerJointData: Codable {
    let x: Double
    let y: Double
    let z: Double? // Added Z coordinate
    let confidence: Double?
}

struct ServerFrameData: Codable {
    let joints: [String: ServerJointData] // Key is joint name string
}

struct VideoAnalysisResponse: Codable {
    let totalFrames: Int
    let jointDataPerFrame: [ServerFrameData] // Will now contain 3D data
    let swingAnalysis: [String: Bool]? // Swing analysis remains 2D based

    enum CodingKeys: String, CodingKey {
        case totalFrames = "total_frames"
        case jointDataPerFrame = "joint_data_per_frame"
        case swingAnalysis = "swing_analysis"
    }
}

struct TechniqueComparisonRequestData: Codable {
    let userVideoFrames: [ServerFrameData] // Will now contain 3D data
    let modelVideoFrames: [ServerFrameData] // Will now contain 3D data
    let dominantSide: String

    enum CodingKeys: String, CodingKey {
        case userVideoFrames = "user_video_frames"
        case modelVideoFrames = "model_video_frames"
        case dominantSide = "dominant_side"
    }
}

// ComparisonResult struct is defined in TechniqueComparisonView.swift
// It's based on the 2D swing analysis from the server.
// struct ComparisonResult: Codable { ... }

class TechniqueAnalysisService {
    // Ensure this URL points to your server's correct IP/domain and port
    private let serverBaseURL = URL(string: "http://115.188.74.78:8000")!

    // Function to upload a single video and get its joint data (now 3D)
    func analyzeVideoByUploading(videoURL: URL, dominantSide: String) -> AnyPublisher<VideoAnalysisResponse, Error> {
        let endpoint = serverBaseURL.appendingPathComponent("/analyze/video_upload/")
        
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var httpBody = Data()
        
        // Add dominant_side part
        httpBody.append("--\(boundary)\r\n".data(using: .utf8)!)
        httpBody.append("Content-Disposition: form-data; name=\"dominant_side\"\r\n\r\n".data(using: .utf8)!)
        httpBody.append("\(dominantSide)\r\n".data(using: .utf8)!)
        
        // Add video file part
        do {
            let videoData = try Data(contentsOf: videoURL)
            let filename = videoURL.lastPathComponent
            // Mimetype for mp4. Consider making this more dynamic if other formats are supported.
            let mimetype = "video/mp4"
            
            httpBody.append("--\(boundary)\r\n".data(using: .utf8)!)
            httpBody.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
            httpBody.append("Content-Type: \(mimetype)\r\n\r\n".data(using: .utf8)!)
            httpBody.append(videoData)
            httpBody.append("\r\n".data(using: .utf8)!)
        } catch {
            print("Error reading video data for upload: \(error)")
            return Fail(error: error).eraseToAnyPublisher()
        }
        
        httpBody.append("--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = httpBody
        
        print("Uploading video for 3D analysis to \(endpoint)... with dominant side: \(dominantSide)")

        return URLSession.shared.dataTaskPublisher(for: request)
            .tryMap { output in
                guard let httpResponse = output.response as? HTTPURLResponse else {
                    print("Invalid server response (not HTTPURLResponse)")
                    throw URLError(.badServerResponse)
                }
                print("Server response status code for 3D video upload: \(httpResponse.statusCode)")
                if !(200...299).contains(httpResponse.statusCode) {
                    if let responseString = String(data: output.data, encoding: .utf8) {
                        print("Server error response (3D upload) [Status: \(httpResponse.statusCode)]: \(responseString)")
                    } else {
                        print("Server error response (3D upload) [Status: \(httpResponse.statusCode)]: No parsable error body.")
                    }
                    throw URLError(.init(rawValue: httpResponse.statusCode), userInfo: [NSLocalizedDescriptionKey: "Server returned status \(httpResponse.statusCode) for 3D video upload"])
                }
                // For debugging the raw response:
                // if let jsonString = String(data: output.data, encoding: .utf8) {
                //    print("Raw server response (3D upload): \(jsonString.prefix(2000))") // Log more characters
                // }
                return output.data
            }
            .decode(type: VideoAnalysisResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }

    // Function to send two sets of 3D joint data to the server for comparison.
    // The server will perform 2D swing analysis on this 3D data.
    func requestTechniqueComparison(
        userFrames: [ServerFrameData], // These now contain 3D data
        modelFrames: [ServerFrameData], // These now contain 3D data
        dominantSide: String
    ) -> AnyPublisher<ComparisonResult, Error> { // ComparisonResult is still based on 2D analysis
        
        let endpoint = serverBaseURL.appendingPathComponent("/analyze/technique_comparison/")
        print("Requesting technique comparison (with 3D frame data input) from: \(endpoint)")

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let requestData = TechniqueComparisonRequestData(
            userVideoFrames: userFrames,
            modelVideoFrames: modelFrames,
            dominantSide: dominantSide
        )

        do {
            request.httpBody = try JSONEncoder().encode(requestData)
            // For debugging the request payload:
            // if let jsonString = String(data: request.httpBody!, encoding: .utf8) {
            //      print("Sending 3D comparison request JSON: \(jsonString.prefix(2000))")
            // }
        } catch {
            print("Error encoding 3D comparison request data: \(error)")
            return Fail(error: error).eraseToAnyPublisher()
        }

        return URLSession.shared.dataTaskPublisher(for: request)
            .tryMap { output in
                guard let httpResponse = output.response as? HTTPURLResponse else {
                    print("Invalid server response (not HTTPURLResponse) for comparison.")
                    throw URLError(.badServerResponse)
                }
                print("Server response status code for 3D input comparison: \(httpResponse.statusCode)")
                if !(200...299).contains(httpResponse.statusCode) {
                    if let responseString = String(data: output.data, encoding: .utf8) {
                        print("Server error response (3D input comparison) [Status: \(httpResponse.statusCode)]: \(responseString)")
                    } else {
                         print("Server error response (3D input comparison) [Status: \(httpResponse.statusCode)]: No parsable error body.")
                    }
                    throw URLError(.init(rawValue: httpResponse.statusCode), userInfo: [NSLocalizedDescriptionKey: "Server returned status \(httpResponse.statusCode) for 3D input comparison"])
                }
                // For debugging raw comparison response:
                // if let jsonString = String(data: output.data, encoding: .utf8) {
                //     print("Raw server response JSON (3D input comparison): \(jsonString)")
                // }
                return output.data
            }
            .decode(type: ComparisonResult.self, decoder: JSONDecoder()) // ComparisonResult structure itself doesn't change yet
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
}

// Helper extension for appending string to Data
extension Data {
    mutating func append(_ string: String) {
        if let data = string.data(using: .utf8) {
            append(data)
        }
    }
}
