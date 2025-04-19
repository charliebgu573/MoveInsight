import SwiftUI
import Vision

// MARK: - Body Connection Structure
// Defines a connection between two body joints for drawing the skeleton
struct BodyConnection: Identifiable {
    let id = UUID()
    let from: VNHumanBodyPoseObservation.JointName
    let to: VNHumanBodyPoseObservation.JointName
}
