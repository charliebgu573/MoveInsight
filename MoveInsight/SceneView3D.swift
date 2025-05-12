import SwiftUI
import Vision
import SceneKit
import simd

// Scene delegate to maintain camera position across updates
class SceneDelegate: NSObject, SCNSceneRendererDelegate, ObservableObject {
    var lastCameraTransform: SCNMatrix4?
    
    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        // Capture camera transform
        if let pointOfView = renderer.pointOfView {
            lastCameraTransform = pointOfView.transform
        }
    }
}

struct SceneView3D: UIViewRepresentable {
    var pose3DBodies: [Pose3DBody]
    var bodyConnections: [BodyConnection]
    @ObservedObject var sceneDelegate = SceneDelegate()
    
    // Define the root joint to use for alignment
    private let rootJoint: VNHumanBodyPoseObservation.JointName = .neck
    
    func makeUIView(context: Context) -> SCNView {
        let sceneView = SCNView()
        let scene = SCNScene()
        
        // Setup camera
        setupCamera(in: scene)
        
        // Setup lighting
        setupLighting(in: scene)
        
        // Add floor grid for better orientation
        let floor = createFloorGrid()
        floor.position = SCNVector3(0, 0, 0)
        scene.rootNode.addChildNode(floor)
        
        // Setup scene
        sceneView.scene = scene
        sceneView.backgroundColor = UIColor.systemBackground
        sceneView.allowsCameraControl = true
        sceneView.showsStatistics = false
        sceneView.delegate = sceneDelegate
        
        return sceneView
    }
    
    func updateUIView(_ uiView: SCNView, context: Context) {
        // Clear existing nodes except for camera, lights, and floor
        uiView.scene?.rootNode.childNodes.forEach { node in
            if node.camera == nil && node.light == nil && node.name != "floor" {
                node.removeFromParentNode()
            }
        }
        
        // Group poses by source
        let primaryPoses = pose3DBodies.filter { $0.videoSource == .primary }
        let secondaryPoses = pose3DBodies.filter { $0.videoSource == .secondary }

        let leftPosition = SIMD3<Float>(0, 0, 0)
        let rightPosition = SIMD3<Float>(0, 0, 0)
        
        // Get primary pose and height
        guard let primaryPose = primaryPoses.first else {
            // If only secondary pose exists, just add it without scaling
            if let secondaryPose = secondaryPoses.first {
                addSkeletonToScene(body: secondaryPose, scene: uiView.scene, position: SIMD3<Float>(0, 0, 0))
                addHeadToScene(body: secondaryPose, scene: uiView.scene, position: SIMD3<Float>(0, 0, 0))
            }
            return
        }
        
        // Calculate primary height to use for scaling
        let primaryHeight = calculateSkeletonHeight(primaryPose)
        
        // Add primary pose (left side)
        addSkeletonToScene(body: primaryPose, scene: uiView.scene, position: leftPosition)
        addHeadToScene(body: primaryPose, scene: uiView.scene, position: leftPosition)
        
        // Add secondary pose (right side) with height matching primary
        if let secondaryPose = secondaryPoses.first {
            // Get scaling factor
            let secondaryHeight = calculateSkeletonHeight(secondaryPose)
            let scaleFactor = primaryHeight / max(secondaryHeight, 0.001)
            
            // Add scaled skeleton
            addSkeletonToScene(body: secondaryPose,
                              scene: uiView.scene,
                              position: rightPosition,
                              scaleFactor: scaleFactor)
            
            // Add head with the same scaling
            addHeadToScene(body: secondaryPose,
                          scene: uiView.scene,
                          position: rightPosition,
                          scaleFactor: scaleFactor)
        }
    }
    
    // Setup camera with initial position - even closer for better visibility
    private func setupCamera(in scene: SCNScene) {
        let camera = SCNCamera()
        camera.usesOrthographicProjection = false
        camera.fieldOfView = 45 // Narrower field of view for more zoom
        camera.zNear = 0.1
        camera.zFar = 100
        
        let cameraNode = SCNNode()
        cameraNode.camera = camera
        
        // Set default camera position if no previous transform exists
        if let lastTransform = sceneDelegate.lastCameraTransform {
            // Use the last camera transform if available
            cameraNode.transform = lastTransform
        } else {
            // Position camera very close to origin for better visibility
            cameraNode.position = SCNVector3(0, 0, 3)
            cameraNode.look(at: SCNVector3(0, 0, 0))
        }
        
        scene.rootNode.addChildNode(cameraNode)
    }
    
    // Setup lighting with ambient and directional lights
    private func setupLighting(in scene: SCNScene) {
        // Add ambient light
        let ambientLight = SCNLight()
        ambientLight.type = .ambient
        ambientLight.intensity = 100
        ambientLight.color = UIColor(white: 0.5, alpha: 1.0)
        
        let ambientLightNode = SCNNode()
        ambientLightNode.light = ambientLight
        scene.rootNode.addChildNode(ambientLightNode)
        
        // Add directional light
        let directionalLight = SCNLight()
        directionalLight.type = .directional
        directionalLight.intensity = 1000
        directionalLight.castsShadow = true
        directionalLight.shadowColor = UIColor.black.withAlphaComponent(0.8)
        
        let directionalLightNode = SCNNode()
        directionalLightNode.light = directionalLight
        directionalLightNode.position = SCNVector3(5, 5, 5)
        directionalLightNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(directionalLightNode)
    }
    
    // Create a floor grid for better orientation
    private func createFloorGrid() -> SCNNode {
        let gridSize: Float = 10
        let gridNode = SCNNode()
        gridNode.name = "floor"
        
        // Create a grid floor for better orientation
        for i in stride(from: -gridSize/2, through: gridSize/2, by: 0.5) {
            // X lines
            let xLine = SCNNode(geometry: SCNBox(width: CGFloat(gridSize), height: 0.001, length: 0.01, chamferRadius: 0))
            xLine.position = SCNVector3(0, 0, i)
            xLine.geometry?.firstMaterial?.diffuse.contents = i == 0 ? UIColor.blue : UIColor.gray.withAlphaComponent(0.5)
            
            // Z lines
            let zLine = SCNNode(geometry: SCNBox(width: 0.01, height: 0.001, length: CGFloat(gridSize), chamferRadius: 0))
            zLine.position = SCNVector3(i, 0, 0)
            zLine.geometry?.firstMaterial?.diffuse.contents = i == 0 ? UIColor.red : UIColor.gray.withAlphaComponent(0.5)
            
            gridNode.addChildNode(xLine)
            gridNode.addChildNode(zLine)
        }
        
        return gridNode
    }
    
    // Add the skeleton without the head
    private func addSkeletonToScene(body: Pose3DBody, scene: SCNScene?, position: SIMD3<Float>? = nil, scaleFactor: Float = 1.0) {
        guard let scene = scene else { return }
        
        // Create a parent node for this skeleton
        let skeletonNode = SCNNode()
        skeletonNode.name = "skeleton"
        
        // Find the lowest joint to place on ground
        let lowestPoint = findLowestPoint(body)
        
        // Calculate the offset to place the skeleton at the specified position
        // and with the lowest point exactly at y=0 (ground level)
        let basePosition = position ?? SIMD3<Float>(0, 0, 0)
        let verticalOffset = -lowestPoint // This will place lowest point at y=0
        
        // Create joint nodes for joints below neck only
        var jointNodes = [VNHumanBodyPoseObservation.JointName: SCNNode]()
        let upperBodyJoints: Set<VNHumanBodyPoseObservation.JointName> = [
            .leftEye, .rightEye, .leftEar, .rightEar, .nose, .neck
        ]
        
        for (jointName, posePoint) in body.joints {
            // Skip joints above shoulders
            if upperBodyJoints.contains(jointName) {
                continue
            }
            
            // Apply offset and scaling to position skeleton correctly
            let adjustedPosition = SIMD3<Float>(
                posePoint.position.x * scaleFactor + basePosition.x,
                posePoint.position.y * scaleFactor + verticalOffset * scaleFactor + basePosition.y,
                posePoint.position.z * scaleFactor + basePosition.z
            )
            
            let jointNode = createJointNode(
                position: adjustedPosition,
                source: body.videoSource
            )
            jointNodes[jointName] = jointNode
            skeletonNode.addChildNode(jointNode)
        }
        
        // Connect joints with bones, ensuring shoulder connection
        connectJoints(joints: jointNodes, in: skeletonNode, connections: bodyConnections, source: body.videoSource)
        
        // Add shoulder connection if exists in joints
        if let leftShoulder = jointNodes[.leftShoulder], let rightShoulder = jointNodes[.rightShoulder] {
            createBoneBetween(
                startPos: leftShoulder.simdPosition,
                endPos: rightShoulder.simdPosition,
                in: skeletonNode,
                source: body.videoSource
            )
        }
        
        // Add the skeleton to the scene
        scene.rootNode.addChildNode(skeletonNode)
    }
    
    // Add just the head
    private func addHeadToScene(body: Pose3DBody, scene: SCNScene?, position: SIMD3<Float>? = nil, scaleFactor: Float = 1.0) {
        guard let scene = scene else { return }
        
        // We need shoulders to place the head properly
        guard let leftShoulderJoint = body.joints[.leftShoulder],
              let rightShoulderJoint = body.joints[.rightShoulder] else {
            return
        }
        
        // Find the lowest joint to place on ground
        let lowestPoint = findLowestPoint(body)
        
        // Calculate the offset for vertical position
        let basePosition = position ?? SIMD3<Float>(0, 0, 0)
        let verticalOffset = -lowestPoint
        
        // Scale and offset shoulder positions
        let leftShoulderPos = SIMD3<Float>(
            leftShoulderJoint.position.x * scaleFactor + basePosition.x,
            leftShoulderJoint.position.y * scaleFactor + verticalOffset * scaleFactor + basePosition.y,
            leftShoulderJoint.position.z * scaleFactor + basePosition.z
        )
        
        let rightShoulderPos = SIMD3<Float>(
            rightShoulderJoint.position.x * scaleFactor + basePosition.x,
            rightShoulderJoint.position.y * scaleFactor + verticalOffset * scaleFactor + basePosition.y,
            rightShoulderJoint.position.z * scaleFactor + basePosition.z
        )
        
        // Calculate center point between shoulders
        let shoulderCenter = (leftShoulderPos + rightShoulderPos) / 2
        
        // Calculate head position just above shoulders
        let headPosition = SIMD3<Float>(
            shoulderCenter.x,                // Center between shoulders (X)
            shoulderCenter.y + 0.1,          // Just slightly above shoulders (Y)
            shoulderCenter.z                 // Same depth as shoulders (Z)
        )
        
        // Create a small head sphere with consistent size between skeletons
        let headRadius = 0.05 // Fixed size regardless of scaling
        let headGeometry = SCNSphere(radius: CGFloat(headRadius))
        let material = SCNMaterial()
        
        // Full opacity head, color based on source
        let color = body.videoSource == .primary ?
            UIColor.systemBlue :
            UIColor.systemRed
            
        material.diffuse.contents = color
        material.specular.contents = UIColor.white
        headGeometry.materials = [material]
        
        // Create head node
        let headNode = SCNNode(geometry: headGeometry)
        headNode.position = SCNVector3(headPosition.x, headPosition.y, headPosition.z)
        
        headNode.opacity = 0.6
        
        // Add the head to the scene
        scene.rootNode.addChildNode(headNode)
    }
    
    private func createJointNode(position: SIMD3<Float>, source: Pose3DBody.VideoSource) -> SCNNode {
        // Create a small sphere for the joint
        let jointGeometry = SCNSphere(radius: 0.01)
        
        let material = SCNMaterial()
        let color = source == .primary ?
            UIColor.systemBlue :
            UIColor.systemRed
        
        material.diffuse.contents = color
        material.specular.contents = UIColor.white
        jointGeometry.materials = [material]
        
        let jointNode = SCNNode(geometry: jointGeometry)
        jointNode.position = SCNVector3(
            position.x,
            position.y,
            position.z
        )
        
        // Apply opacity to entire node if secondary
        if source == .secondary {
            jointNode.opacity = 0.5
        }
        
        return jointNode
    }
    
    private func connectJoints(joints: [VNHumanBodyPoseObservation.JointName: SCNNode],
                              in parentNode: SCNNode,
                              connections: [BodyConnection],
                              source: Pose3DBody.VideoSource) {
        
        for connection in connections {
            // Skip connections involving neck
            if connection.from == .neck || connection.to == .neck {
                continue
            }
            
            guard let startNode = joints[connection.from],
                  let endNode = joints[connection.to] else {
                continue
            }
            
            let startPos = startNode.simdPosition
            let endPos = endNode.simdPosition
            
            // Create a bone between the joints
            createBoneBetween(startPos: startPos,
                             endPos: endPos,
                             in: parentNode,
                             source: source)
        }
    }
    
    private func createBoneBetween(startPos: SIMD3<Float>,
                                  endPos: SIMD3<Float>,
                                  in parentNode: SCNNode,
                                  source: Pose3DBody.VideoSource) {
        
        // Calculate the midpoint between start and end positions
        let midPoint = (startPos + endPos) / 2
        
        // Calculate the distance between points
        let distance = simd_distance(startPos, endPos)
        
        // Create a cylinder for the bone
        let boneGeometry = SCNCylinder(radius: 0.01, height: CGFloat(distance))
        let material = SCNMaterial()
        
        // Full opacity for bones
        let color = source == .primary ?
            UIColor.systemBlue :
            UIColor.systemRed
        
        material.diffuse.contents = color
        boneGeometry.materials = [material]
        
        let boneNode = SCNNode(geometry: boneGeometry)
        
        // Position the bone at the midpoint
        boneNode.position = SCNVector3(midPoint.x, midPoint.y, midPoint.z)
        
        // Calculate the orientation to point from start to end
        let direction = simd_normalize(endPos - startPos)
        let upVector = SIMD3<Float>(0, 1, 0)
        
        // If direction is parallel to up vector, use a different up vector
        let rotationAxis: SIMD3<Float>
        if abs(simd_dot(direction, upVector)) > 0.999 {
            rotationAxis = simd_cross(SIMD3<Float>(1, 0, 0), direction)
        } else {
            rotationAxis = simd_cross(upVector, direction)
        }
        
        let rotationAngle = acos(simd_dot(upVector, direction))
        
        if simd_length(rotationAxis) > 1e-5 {
            let normalizedRotationAxis = simd_normalize(rotationAxis)
            let quaternion = simd_quatf(angle: rotationAngle, axis: normalizedRotationAxis)
            boneNode.simdOrientation = quaternion
        }

        boneNode.opacity = 0.6
        
        parentNode.addChildNode(boneNode)
    }
    
    // Find the lowest point in the skeleton to place on ground
    private func findLowestPoint(_ body: Pose3DBody) -> Float {
        var lowestY = Float.greatestFiniteMagnitude
        
        for (_, joint) in body.joints {
            lowestY = min(lowestY, joint.position.y)
        }
        
        // If no joints found, return 0
        return lowestY != Float.greatestFiniteMagnitude ? lowestY : 0
    }
    
    // Calculate the height of a skeleton (from lowest to highest point)
    private func calculateSkeletonHeight(_ body: Pose3DBody) -> Float {
        var lowest = Float.greatestFiniteMagnitude
        var highest = -Float.greatestFiniteMagnitude
        
        for (_, joint) in body.joints {
            lowest = min(lowest, joint.position.y)
            highest = max(highest, joint.position.y)
        }
        
        return highest - lowest
    }
}
