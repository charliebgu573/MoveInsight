import Foundation
import Vision
import Combine

class TechniqueAnalysisService {
    private let serverURL = URL(string: "http://115.188.74.78:8000")!
    
    struct JointData: Codable {
        let x: Float
        let y: Float
        let confidence: Float
    }
    
    struct FrameJointsData: Codable {
        let joints: [String: [JointData]]
        let dominantSide: String
        
        enum CodingKeys: String, CodingKey {
            case joints
            case dominantSide = "dominant_side"
        }
    }
    
    func compareTechniques(
        userViewModel: VideoPlayerViewModel,
        modelViewModel: VideoPlayerViewModel,
        dominantSide: String = "Right"
    ) -> AnyPublisher<ComparisonResult, Error> {
        // Use getAllPoses to get accumulated poses over time
        let userJoints = extractJointsForServer(from: userViewModel.getAllPoses())
        let modelJoints = extractJointsForServer(from: modelViewModel.getAllPoses())
        
        // Debug info
        print("Sending analysis with \(userViewModel.getAllPoses().count) user frames and \(modelViewModel.getAllPoses().count) model frames")
        if let firstJoint = userJoints.values.first {
            print("First joint has \(firstJoint.count) frames of data")
        }
        
        let userFrameJoints = FrameJointsData(joints: userJoints, dominantSide: dominantSide)
        let modelFrameJoints = FrameJointsData(joints: modelJoints, dominantSide: dominantSide)
        
        let comparisonData = ["user": userFrameJoints, "reference": modelFrameJoints]
        
        return sendRequest(to: "/analyze/comparison", body: comparisonData)
    }
    
    private func extractJointsForServer(from poses: [[VNHumanBodyPoseObservation.JointName: CGPoint]]) -> [String: [JointData]] {
        var result: [String: [JointData]] = [:]
        
        // Log the number of frames we're processing
        print("Processing \(poses.count) frames for analysis")
        
        // Get all unique joint names that appear across all frames
        var allJointNames = Set<String>()
        
        // Map Vision API joint names to Python expected names
        let jointNameMapping: [VNHumanBodyPoseObservation.JointName: String] = [
            .nose: "Nose",
            .leftShoulder: "LeftShoulder",
            .rightShoulder: "RightShoulder",
            .leftElbow: "LeftElbow",
            .rightElbow: "RightElbow",
            .leftWrist: "LeftWrist",
            .rightWrist: "RightWrist",
            .leftHip: "LeftHip",
            .rightHip: "RightHip",
            .leftKnee: "LeftKnee",
            .rightKnee: "RightKnee",
            .leftAnkle: "LeftAnkle",
            .rightAnkle: "RightAnkle"
        ]
        
        // Add special mappings for heel and toe which might not directly map to Vision API
        let specialMappings: [VNHumanBodyPoseObservation.JointName: [String]] = [
            .leftAnkle: ["LeftHeel", "LeftToe"],
            .rightAnkle: ["RightHeel", "RightToe"]
        ]
        
        // Initialize arrays for each joint type
        for joint in jointNameMapping.values {
            result[joint] = []
            allJointNames.insert(joint)
        }
        
        // Add heel and toe (special cases)
        for specialJoints in specialMappings.values {
            for joint in specialJoints {
                result[joint] = []
                allJointNames.insert(joint)
            }
        }
        
        // CRITICAL: This is the key fix - Go through EACH FRAME'S poses and extract joint data
        for frameIndex in 0..<poses.count {
            let pose = poses[frameIndex]
            
            // Add regular joints
            for (visionJoint, serverJoint) in jointNameMapping {
                if let point = pose[visionJoint] {
                    result[serverJoint]?.append(JointData(x: Float(point.x), y: Float(point.y), confidence: 1.0))
                } else {
                    // If joint missing in this frame, use placeholder
                    result[serverJoint]?.append(JointData(x: 0, y: 0, confidence: 0))
                }
            }
            
            // Handle special cases (heel and toe)
            for (ankleJoint, specialJoints) in specialMappings {
                if let anklePoint = pose[ankleJoint] {
                    // For heel, offset backward from ankle
                    let heel = CGPoint(x: anklePoint.x, y: anklePoint.y + 0.03)
                    result[specialJoints[0]]?.append(JointData(x: Float(heel.x), y: Float(heel.y), confidence: 0.8))
                    
                    // For toe, offset forward from ankle
                    let toe = CGPoint(x: anklePoint.x, y: anklePoint.y - 0.05)
                    result[specialJoints[1]]?.append(JointData(x: Float(toe.x), y: Float(toe.y), confidence: 0.8))
                } else {
                    // If ankle not found, add placeholders
                    for joint in specialJoints {
                        result[joint]?.append(JointData(x: 0, y: 0, confidence: 0))
                    }
                }
            }
        }
        
        // Debug logging
        for (joint, frames) in result {
            print("Joint \(joint): \(frames.count) frames")
        }
        
        // Ensure there's enough data for analysis - minimum 10 frames
        if let firstJoint = result.values.first, firstJoint.count < 10 {
            print("WARNING: Not enough frames for analysis. Found only \(firstJoint.count) frames, minimum 10 recommended")
        }
        
        // Filter out empty arrays
        return result.filter { !$0.value.isEmpty }
    }
    
    private func sendRequest<Input: Encodable, Output: Decodable>(to endpoint: String, body: Input) -> AnyPublisher<Output, Error> {
        let url = serverURL.appendingPathComponent(endpoint)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            let encoder = JSONEncoder()
            request.httpBody = try encoder.encode(body)
        } catch {
            return Fail(error: error).eraseToAnyPublisher()
        }
        
        return URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: Output.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .eraseToAnyPublisher()
    }
}
