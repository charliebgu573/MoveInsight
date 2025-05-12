import Vision
import simd
import CoreVideo

class Pose3DProcessor {
    
    // Convert 2D pose point with depth to 3D
    func convert2DPoseToWorldSpace(
        pose: [VNHumanBodyPoseObservation.JointName: CGPoint],
        depthBuffer: CVPixelBuffer,
        videoSize: CGSize
    ) -> [VNHumanBodyPoseObservation.JointName: SIMD3<Float>] {
        var pose3D: [VNHumanBodyPoseObservation.JointName: SIMD3<Float>] = [:]
        
        // Get dimensions of depth buffer
        let depthWidth = CVPixelBufferGetWidth(depthBuffer)
        let depthHeight = CVPixelBufferGetHeight(depthBuffer)
        
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        
        // Get base address assuming it's grayscale16half format
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthBuffer) else {
            return [:]
        }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthBuffer)
        
        for (joint, point) in pose {
            // Convert normalized coordinates to pixel coordinates in the depth buffer
            let depthX = Int(point.x * CGFloat(depthWidth))
            let depthY = Int(point.y * CGFloat(depthHeight))
            
            // Ensure coordinates are within bounds
            if depthX >= 0 && depthX < depthWidth && depthY >= 0 && depthY < depthHeight {
                // Get depth value (16-bit half float)
                let pixelAddress = baseAddress.advanced(by: depthY * bytesPerRow + depthX * 2)
                let halfFloat = pixelAddress.assumingMemoryBound(to: UInt16.self).pointee
                
                // Convert half float to float
                let depth = convertHalfToFloat(halfFloat)
                
                // Create 3D point: x, y from 2D pose, z from depth
                let x = Float(point.x * 2 - 1) // Convert 0-1 to -1 to 1
                let y = Float(1 - point.y * 2) // Convert 0-1 to 1 to -1 (y-axis flipped in 3D)
                let z = depth
                
                pose3D[joint] = SIMD3<Float>(x, y, z)
            }
        }
        
        return pose3D
    }
    
    // Helper to convert UInt16 half float to Float
    private func convertHalfToFloat(_ half: UInt16) -> Float {
        // Using a more direct bit manipulation approach
        let sign = (half & 0x8000) != 0
        let exponent = Int((half & 0x7C00) >> 10)
        let fraction = half & 0x03FF
        
        // Handle special cases
        if exponent == 0 {
            if fraction == 0 {
                return sign ? -0.0 : 0.0 // Zero
            } else {
                // Denormalized number
                var result = Float(fraction) / Float(1024)
                result *= pow(2.0, -14.0)
                return sign ? -result : result
            }
        } else if exponent == 31 {
            if fraction == 0 {
                return sign ? Float.infinity : Float.infinity
            } else {
                return Float.nan
            }
        }
        
        // Normalized number
        var result = Float(1 + Float(fraction) / 1024.0)
        result *= pow(2.0, Float(exponent - 15))
        return sign ? -result : result
    }
    
    // Processing multiple poses from a frame
    func process(
        poses: [[VNHumanBodyPoseObservation.JointName: CGPoint]],
        depthBuffer: CVPixelBuffer,
        videoSize: CGSize
    ) -> [[VNHumanBodyPoseObservation.JointName: SIMD3<Float>]] {
        return poses.map { pose in
            convert2DPoseToWorldSpace(
                pose: pose,
                depthBuffer: depthBuffer,
                videoSize: videoSize
            )
        }
    }
}
