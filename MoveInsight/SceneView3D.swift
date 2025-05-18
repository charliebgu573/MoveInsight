// MoveInsight/SceneView3D.swift
import SwiftUI
import SceneKit
import simd // For SIMD3<Float>

// SceneKitViewDelegate: Manages SceneKit renderer updates and camera state.
class SceneKitViewDelegate: NSObject, SCNSceneRendererDelegate, ObservableObject {
    @Published var lastCameraTransform: SCNMatrix4?

    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        if let pointOfView = renderer.pointOfView {
            DispatchQueue.main.async {
                if let lastTransform = self.lastCameraTransform {
                    if !SCNMatrix4EqualToMatrix4(lastTransform, pointOfView.transform) {
                        self.lastCameraTransform = pointOfView.transform
                    }
                } else {
                    self.lastCameraTransform = pointOfView.transform
                }
            }
        }
    }
}

// SceneView3D: A SwiftUI UIViewRepresentable for displaying 3D skeletons.
struct SceneView3D: UIViewRepresentable {
    let userPose3D: [String: SIMD3<Float>]?
    let modelPose3D: [String: SIMD3<Float>]?
    let bodyConnections: [BodyConnection3D]

    // zScale: Adjusts the depth perception. Tune as needed.
    private let zScale: Float = 0.7
    private let skeletonNodeName = "skeletonRootNode"
    // floorLevelY: The Y-coordinate where the floor is placed. Skeletons will stand on this.
    private let floorLevelY: Float = -0.75
    // skeletonHeightScale: Multiplies the normalized Y-coordinates to give skeletons a reasonable height in the scene.
    private let skeletonHeightScale: Float = 1.5 // Adjust this to make skeletons taller or shorter

    @ObservedObject var sceneDelegate: SceneKitViewDelegate

    func makeUIView(context: Context) -> SCNView {
        let sceneView = SCNView()
        sceneView.scene = SCNScene()

        setupCamera(in: sceneView.scene!, view: sceneView)
        setupLighting(in: sceneView.scene!)
        setupFloorWithGrid(in: sceneView.scene!) // Updated to add a grid pattern
        
        sceneView.backgroundColor = UIColor.systemGray5
        sceneView.allowsCameraControl = true
        sceneView.showsStatistics = false
        sceneView.delegate = sceneDelegate
        sceneView.antialiasingMode = .multisampling4X
        
        return sceneView
    }
    
    func updateUIView(_ uiView: SCNView, context: Context) {
        guard let scene = uiView.scene else { return }

        // Remove previously drawn skeletons
        scene.rootNode.childNodes { (node, _) -> Bool in
            node.name == skeletonNodeName
        }.forEach { $0.removeFromParentNode() }

        // Add user skeleton
        if let userPose = userPose3D, !userPose.isEmpty {
            addSkeletonToScene(
                pose: userPose,
                scene: scene,
                color: .systemBlue,
                baseOffset: SIMD3<Float>(-0.4, 0, 0) // X-offset for side-by-side placement
            )
        }
        
        // Add model skeleton
        if let modelPose = modelPose3D, !modelPose.isEmpty {
            addSkeletonToScene(
                pose: modelPose,
                scene: scene,
                color: .systemRed,
                baseOffset: SIMD3<Float>(0.4, 0, 0) // X-offset for side-by-side placement
            )
        }
    }
    
    private func setupCamera(in scene: SCNScene, view: SCNView) {
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.zNear = 0.1
        cameraNode.camera?.zFar = 100
        cameraNode.camera?.fieldOfView = 50

        if let transform = sceneDelegate.lastCameraTransform {
            cameraNode.transform = transform
        } else {
            // Initial camera position: looking at origin (0,0,0), slightly elevated, positive Z.
            cameraNode.position = SCNVector3(x: 0, y: 0.3, z: 2.8) // Adjusted Z for a bit more distance
            cameraNode.look(at: SCNVector3(0, 0, 0))
        }
        scene.rootNode.addChildNode(cameraNode)
    }
    
    private func setupLighting(in scene: SCNScene) {
        // Ambient light for overall illumination
        let ambientLightNode = SCNNode()
        ambientLightNode.light = SCNLight()
        ambientLightNode.light!.type = .ambient
        ambientLightNode.light!.color = UIColor(white: 0.7, alpha: 1.0)
        scene.rootNode.addChildNode(ambientLightNode)
        
        // Directional light for highlights and shadows
        let directionalLightNode = SCNNode()
        directionalLightNode.light = SCNLight()
        directionalLightNode.light!.type = .directional
        directionalLightNode.light!.color = UIColor(white: 0.8, alpha: 1.0)
        directionalLightNode.light!.castsShadow = true
        directionalLightNode.light!.shadowMode = .deferred
        directionalLightNode.light!.shadowColor = UIColor.black.withAlphaComponent(0.4)
        directionalLightNode.light!.shadowSampleCount = 16
        directionalLightNode.light!.shadowRadius = 3.0
        directionalLightNode.position = SCNVector3(x: -1.5, y: 2.5, z: 2)
        directionalLightNode.look(at: SCNVector3(0,0,0))
        scene.rootNode.addChildNode(directionalLightNode)
    }

    // Sets up the floor with a checkerboard pattern to simulate a grid.
    private func setupFloorWithGrid(in scene: SCNScene) {
        let floorGeometry = SCNFloor()
        floorGeometry.reflectivity = 0.05
        
        // Create a checkerboard material for the floor
        let floorMaterial = SCNMaterial()
        if #available(iOS 13.0, *) { // Check for iOS 13 availability for CIFilter
            let checkerboard = CIFilter(name: "CICheckerboardGenerator")!
            checkerboard.setValue(CIColor.gray, forKey: "inputColor0") // Color for light squares
            checkerboard.setValue(CIColor.black, forKey: "inputColor1") // Color for dark squares
            checkerboard.setValue(80, forKey: "inputWidth") // Width of each square in the pattern
            checkerboard.setValue(CIVector(x: 0, y: 0), forKey: "inputCenter")
            if let ciImage = checkerboard.outputImage {
                floorMaterial.diffuse.contents = ciImage
                // Adjust texture wrapping and scaling if needed
                floorMaterial.diffuse.wrapS = .repeat
                floorMaterial.diffuse.wrapT = .repeat
                // Scale the texture to make the grid appear reasonably sized on the floor.
                // A transform of SCNMatrix4MakeScale(10, 10, 10) means the texture repeats 10 times.
                floorMaterial.diffuse.contentsTransform = SCNMatrix4MakeScale(20, 20, 1) // Repeat texture more for smaller grid cells
            } else {
                floorMaterial.diffuse.contents = UIColor.systemGray4 // Fallback color
            }
        } else {
            floorMaterial.diffuse.contents = UIColor.systemGray4 // Fallback for older iOS
        }
        floorMaterial.lightingModel = .physicallyBased
        floorGeometry.materials = [floorMaterial]
        
        let floorNode = SCNNode(geometry: floorGeometry)
        floorNode.position = SCNVector3(0, floorLevelY, 0)
        scene.rootNode.addChildNode(floorNode)
        
        // Note: For a more distinct line grid, you would typically use an image texture
        // with lines drawn on it, e.g., UIImage(named: "grid_texture.png").
        // The checkerboard provides a basic grid-like pattern.
    }
    
    // Adds a skeleton to the scene, adjusting its Y position to stand on the floor.
    private func addSkeletonToScene(pose: [String: SIMD3<Float>], scene: SCNScene, color: UIColor, baseOffset: SIMD3<Float>) {
        let skeletonRoot = SCNNode()
        skeletonRoot.name = skeletonNodeName
        
        var jointNodes: [String: SCNNode] = [:]
        var minYInSkeletonSpace: Float = Float.greatestFiniteMagnitude

        // Store transformed joint positions relative to the skeleton's own origin
        var relativeJointPositions: [String: SCNVector3] = [:]

        // First pass: Transform coordinates (including Y-inversion) and find the minimum Y.
        for (jointName, position3D) in pose {
            // Assuming position3D.x and position3D.y are normalized (0-1)
            // X: Center it if needed, e.g., (position3D.x - 0.5) * someXScale
            let transformedX = (position3D.x - 0.5) * skeletonHeightScale // Center X and scale
            
            // Y: Invert (0=top becomes 1=top_of_skeleton_height) and scale.
            // (1.0 - position3D.y) makes 0 bottom of normalized range, 1 top.
            // Then scale by skeletonHeightScale.
            let transformedY = (1.0 - position3D.y) * skeletonHeightScale
            
            let transformedZ = position3D.z * zScale
            
            let relativePos = SCNVector3(transformedX, transformedY, transformedZ)
            relativeJointPositions[jointName] = relativePos
            minYInSkeletonSpace = min(minYInSkeletonSpace, transformedY)
        }

        // If no valid minY was found (e.g., empty pose), default to 0 to avoid issues.
        if minYInSkeletonSpace == Float.greatestFiniteMagnitude {
            minYInSkeletonSpace = 0
        }
        
        // Calculate the Y adjustment needed to place the skeleton's lowest point (minYInSkeletonSpace) on the floorLevelY.
        let yShiftToFloor = floorLevelY - minYInSkeletonSpace
        
        // Set the final position for the skeleton root node.
        // The Y position includes the base offset (usually 0 for Y) and the calculated shift.
        skeletonRoot.position = SCNVector3(baseOffset.x, yShiftToFloor + baseOffset.y, baseOffset.z)
        scene.rootNode.addChildNode(skeletonRoot)

        // Second pass: Create SCNNode for each joint using its relative position and add to skeletonRoot.
        for (jointName, relativePos) in relativeJointPositions {
            let jointNode = createJointSphereNode(radius: 0.025, color: color)
            jointNode.position = relativePos // Position is relative to skeletonRoot
            skeletonRoot.addChildNode(jointNode)
            jointNodes[jointName] = jointNode // Store for bone creation
        }
        
        // Create cylinder nodes for bones, connecting the joint spheres.
        // These are also relative to skeletonRoot.
        for connection in bodyConnections {
            guard let fromNode = jointNodes[connection.from],
                  let toNode = jointNodes[connection.to] else {
                continue
            }
            let boneNode = createBoneCylinderNode(from: fromNode.position, to: toNode.position, radius: 0.012, color: color)
            skeletonRoot.addChildNode(boneNode)
        }
    }
    
    private func createJointSphereNode(radius: CGFloat, color: UIColor) -> SCNNode {
        let sphere = SCNSphere(radius: radius)
        let material = SCNMaterial()
        material.diffuse.contents = color
        material.lightingModel = .phong
        sphere.materials = [material]
        return SCNNode(geometry: sphere)
    }
    
    private func createBoneCylinderNode(from startPoint: SCNVector3, to endPoint: SCNVector3, radius: CGFloat, color: UIColor) -> SCNNode {
        let height = SCNVector3.distance(vectorStart: startPoint, vectorEnd: endPoint)
        guard height > 0.001 else { return SCNNode() }

        let cylinder = SCNCylinder(radius: radius, height: CGFloat(height))
        let material = SCNMaterial()
        material.diffuse.contents = color
        material.lightingModel = .phong
        cylinder.materials = [material]
        
        let boneNode = SCNNode(geometry: cylinder)
        
        boneNode.position = SCNVector3(
            (startPoint.x + endPoint.x) / 2,
            (startPoint.y + endPoint.y) / 2,
            (startPoint.z + endPoint.z) / 2
        )
        
        let directionVector = endPoint - startPoint
        let yAxis = SCNVector3(0, 1, 0)
        
        let rotationAxis = yAxis.cross(directionVector).normalized()
        var angle = acos(yAxis.dot(directionVector) / (yAxis.length() * directionVector.length()))

        if rotationAxis.length() < 0.001 {
            if directionVector.y < 0 {
                angle = .pi
                 boneNode.rotation = SCNVector4(1, 0, 0, Float.pi)
            } else {
                boneNode.rotation = SCNVector4(0,0,0,0)
            }
        } else if !rotationAxis.x.isNaN && !rotationAxis.y.isNaN && !rotationAxis.z.isNaN && !angle.isNaN {
             boneNode.rotation = SCNVector4(rotationAxis.x, rotationAxis.y, rotationAxis.z, angle)
        }
        return boneNode
    }
}

// Helper SCNVector3 extensions
extension SCNVector3 {
    static func distance(vectorStart: SCNVector3, vectorEnd: SCNVector3) -> Float {
        let dx = vectorEnd.x - vectorStart.x; let dy = vectorEnd.y - vectorStart.y; let dz = vectorEnd.z - vectorStart.z
        return sqrt(dx*dx + dy*dy + dz*dz)
    }
    func length() -> Float { return sqrt(x*x + y*y + z*z) }
    func normalized() -> SCNVector3 {
        let len = length(); if len == 0 { return SCNVector3(0,0,0) }
        return SCNVector3(x/len, y/len, z/len)
    }
    func cross(_ vector: SCNVector3) -> SCNVector3 {
        return SCNVector3(y * vector.z - z * vector.y, z * vector.x - x * vector.z, x * vector.y - y * vector.x)
    }
    func dot(_ vector: SCNVector3) -> Float { return x * vector.x + y * vector.y + z * vector.z }
}
func -(left: SCNVector3, right: SCNVector3) -> SCNVector3 {
    return SCNVector3Make(left.x - right.x, left.y - right.y, left.z - right.z)
}
