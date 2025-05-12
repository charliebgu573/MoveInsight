import CoreML
import Vision
import AVFoundation
import Accelerate
import UIKit

class DepthEstimationService {
    private var depthModel: MLModel?
    
    init() {
        do {
            let modelURL = Bundle.main.url(forResource: "DepthAnythingV2SmallF16", withExtension: "mlmodelc")!
            let config = MLModelConfiguration()
            self.depthModel = try MLModel(contentsOf: modelURL, configuration: config)
            print("Loaded depth model successfully.")
        } catch {
            print("Failed to load depth model: \(error)")
        }
    }
    
    func estimateDepth(from pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        guard let model = depthModel else {
            print("Depth model not loaded")
            return nil
        }
        
        do {
            // Resize according to model requirements
            let resizedPixelBuffer = resizePixelBufferToRequiredSize(pixelBuffer)
            let width = CVPixelBufferGetWidth(resizedPixelBuffer)
            let height = CVPixelBufferGetHeight(resizedPixelBuffer)
            print("Resized to \(width) x \(height)")
            
            // Create MLFeatureValue for the image
            let imageFeatureValue = MLFeatureValue(pixelBuffer: resizedPixelBuffer)
            
            // Create input feature dictionary
            let inputFeatures = try MLDictionaryFeatureProvider(dictionary: ["image": imageFeatureValue])
            
            // Make prediction
            let outputFeatures = try model.prediction(from: inputFeatures)
            
            // Extract the depth map
            if let depthFeatureValue = outputFeatures.featureValue(for: "depth"),
               let depthPixelBuffer = depthFeatureValue.imageBufferValue {
                return depthPixelBuffer
            } else {
                print("Failed to extract depth output")
                return nil
            }
        } catch {
            print("Depth estimation failed: \(error)")
            return nil
        }
    }
    
    // Resize according to specific model requirements
    private func resizePixelBufferToRequiredSize(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        // Target dimensions
        // Width should be 518, height should be 392 (as per your example)
        let targetWidth = 518
        let targetHeight = 392
        
        // Create a CIImage from the pixel buffer
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Scale the image
        let scaleX = CGFloat(targetWidth) / CGFloat(width)
        let scaleY = CGFloat(targetHeight) / CGFloat(height)
        let scaledImage = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        // Create a new pixel buffer
        var newPixelBuffer: CVPixelBuffer?
        let options = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        
        CVPixelBufferCreate(kCFAllocatorDefault,
                           targetWidth,
                           targetHeight,
                           kCVPixelFormatType_32BGRA,
                           options,
                           &newPixelBuffer)
        
        guard let outputPixelBuffer = newPixelBuffer else {
            return pixelBuffer // Return original if resize fails
        }
        
        // Render the scaled CIImage to the new pixel buffer
        let context = CIContext()
        context.render(scaledImage, to: outputPixelBuffer)
        
        return outputPixelBuffer
    }
    
    // Convert depth pixel buffer to UIImage for visualization
    func depthMapToImage(from depthBuffer: CVPixelBuffer) -> UIImage? {
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        
        // For Grayscale16Half format, we need to convert it to a format UIImage can use
        let width = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)
        
        // Create a CIImage from the depth buffer
        var ciImage: CIImage?
        if let colorSpace = CGColorSpace(name: CGColorSpace.linearGray) {
            ciImage = CIImage(cvImageBuffer: depthBuffer, options: [.colorSpace: colorSpace])
        } else {
            ciImage = CIImage(cvImageBuffer: depthBuffer)
        }
        
        guard let image = ciImage else { return nil }
        
        // Apply visualization filters to make depth more visible
        let normalizedImage = image.applyingFilter("CIColorControls", parameters: [
            kCIInputContrastKey: 1.5,
            kCIInputBrightnessKey: 0.2
        ])
        
        // Create a colored version using the turbo colormap
        let coloredImage = normalizedImage.applyingFilter("CIColorMap", parameters: [
            "inputGradientImage": createTurboColormap()
        ])
        
        // Convert to UIImage
        let context = CIContext()
        if let cgImage = context.createCGImage(coloredImage, from: coloredImage.extent) {
            return UIImage(cgImage: cgImage)
        }
        
        return nil
    }
    
    // Create a turbo colormap for better depth visualization
    private func createTurboColormap() -> CIImage {
        let colors: [(CGFloat, CGFloat, CGFloat)] = [
            (0.18995, 0.07176, 0.23217), // Dark blue
            (0.19483, 0.22800, 0.47607), // Blue
            (0.01555, 0.44879, 0.69486), // Light blue
            (0.12943, 0.65563, 0.67862), // Cyan
            (0.33486, 0.81853, 0.39915), // Green
            (0.66724, 0.88581, 0.25420), // Yellow-green
            (0.90480, 0.91255, 0.10421), // Yellow
            (0.99796, 0.68829, 0.02774), // Orange
            (0.95909, 0.37228, 0.01549), // Red-orange
            (0.73683, 0.01779, 0.01820)  // Dark red
        ]
        
        let width = 256
        let gradientImage = CIImage(color: CIColor(red: 0, green: 0, blue: 0)).cropped(to: CGRect(x: 0, y: 0, width: width, height: 1))
        
        var result = gradientImage
        
        // Create gradient segments
        for i in 0..<(colors.count-1) {
            let startColor = CIColor(red: colors[i].0, green: colors[i].1, blue: colors[i].2)
            let endColor = CIColor(red: colors[i+1].0, green: colors[i+1].1, blue: colors[i+1].2)
            
            let startX = (width * i) / (colors.count - 1)
            let endX = (width * (i + 1)) / (colors.count - 1)
            
            let gradient = CIFilter(name: "CILinearGradient", parameters: [
                "inputPoint0": CIVector(x: CGFloat(startX), y: 0),
                "inputPoint1": CIVector(x: CGFloat(endX), y: 0),
                "inputColor0": startColor,
                "inputColor1": endColor
            ])!.outputImage!
            
            // Blend with previous result
            result = result.applyingFilter("CISourceOverCompositing", parameters: [
                "inputBackgroundImage": gradient
            ])
        }
        
        return result
    }
}
