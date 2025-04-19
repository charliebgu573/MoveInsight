import SwiftUI
import AVKit
import Vision
import Combine

// MARK: - Video Player View Model
class VideoPlayerViewModel: ObservableObject {
    let videoURL: URL
    let player: AVPlayer
    let asset: AVAsset

    // Published properties to update the UI
    @Published var poses: [[VNHumanBodyPoseObservation.JointName: CGPoint]] = []
    @Published var bodyConnections: [BodyConnection] = []
    @Published var isVideoReady = false
    @Published var isPlaying = false
    @Published var videoOrientation: CGImagePropertyOrientation = .up

    private var playerItemVideoOutput: AVPlayerItemVideoOutput?
    private var displayLink: CADisplayLink?
    private var playerItemStatusObserver: NSKeyValueObservation?
    private var cancellables = Set<AnyCancellable>()

    // Vision request handler
    private let visionSequenceHandler = VNSequenceRequestHandler()
    // Configure the request for multi-person detection
    private let bodyPoseRequest: VNDetectHumanBodyPoseRequest = {
        let request = VNDetectHumanBodyPoseRequest()
        return request
    }()

    init(videoURL: URL) {
        self.videoURL = videoURL
        self.asset = AVAsset(url: videoURL)
        self.player = AVPlayer()

        print("VideoPlayerViewModel initialized with URL: \(videoURL.path)")

        setupBodyConnections()
        prepareVideoPlayback()
    }

    // Define the connections for the skeleton overlay
    private func setupBodyConnections() {
        bodyConnections = [
            .init(from: .nose, to: .neck),
            .init(from: .neck, to: .rightShoulder),
            .init(from: .neck, to: .leftShoulder),
            .init(from: .rightShoulder, to: .rightHip),
            .init(from: .leftShoulder, to: .leftHip),
            .init(from: .rightHip, to: .leftHip),
            .init(from: .rightShoulder, to: .rightElbow),
            .init(from: .rightElbow, to: .rightWrist),
            .init(from: .leftShoulder, to: .leftElbow),
            .init(from: .leftElbow, to: .leftWrist),
            .init(from: .rightHip, to: .rightKnee),
            .init(from: .rightKnee, to: .rightAnkle),
            .init(from: .leftHip, to: .leftKnee),
            .init(from: .leftKnee, to: .leftAnkle)
        ]
    }

    // Prepare the AVPlayerItem and related components
    private func prepareVideoPlayback() {
        // Asynchronously load asset properties (tracks and duration)
        Task {
            do {
                // Load tracks to get orientation
                let tracks = try await asset.load(.tracks)
                
                // Find the video track and determine orientation
                if let videoTrack = tracks.first(where: { $0.mediaType == .video }) {
                    let transform = try await videoTrack.load(.preferredTransform)
                    let orientation = orientation(from: transform)
                    // Update orientation on the main thread
                    await MainActor.run {
                        print("Determined video orientation: \(orientation.rawValue)")
                        self.videoOrientation = orientation
                        // Now that orientation is known, create player item and setup player
                        self.setupPlayerItemAndObservers()
                    }
                } else {
                    print("Error: No video track found in asset.")
                    await MainActor.run { self.isVideoReady = false }
                }
            } catch {
                print("Error loading asset tracks or transform: \(error)")
                await MainActor.run { self.isVideoReady = false }
            }
        }
    }
    
    // Sets up the player item and observers AFTER orientation is determined
    private func setupPlayerItemAndObservers() {
        let playerItem = AVPlayerItem(asset: asset)
        
        // 1. Observe PlayerItem Status
        playerItemStatusObserver = playerItem.observe(\.status, options: [.new, .initial]) { [weak self] item, _ in
            guard let self = self else { return }
            DispatchQueue.main.async {
                print("PlayerItem status changed: \(item.status.rawValue)")
                switch item.status {
                case .readyToPlay:
                    print("PlayerItem is ready to play.")
                    self.isVideoReady = true
                    self.setupPlayerOutput(for: item)
                    self.setupDisplayLink()
                case .failed:
                    print("PlayerItem failed to load: \(item.error?.localizedDescription ?? "Unknown error")")
                    self.isVideoReady = false
                case .unknown:
                    print("PlayerItem status is unknown.")
                    self.isVideoReady = false
                @unknown default:
                    self.isVideoReady = false
                }
            }
        }

        // 2. Observe Playback End
        NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime, object: playerItem, queue: .main) { [weak self] _ in
            print("Video finished playing.")
            self?.isPlaying = false
            self?.player.seek(to: .zero)
        }
        
        // Replace the player's current item
        player.replaceCurrentItem(with: playerItem)
        
        // Ensure audio plays if present
        player.volume = 1.0
        player.allowsExternalPlayback = true
    }

    // Setup the AVPlayerItemVideoOutput to grab frames
    private func setupPlayerOutput(for item: AVPlayerItem) {
        // Check if output already exists for this item
        if item.outputs.contains(where: { $0 is AVPlayerItemVideoOutput }) {
            print("AVPlayerItemVideoOutput already added.")
            playerItemVideoOutput = item.outputs.first(where: { $0 is AVPlayerItemVideoOutput }) as? AVPlayerItemVideoOutput
            return
        }
        
        let pixelBufferAttributes = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ] as [String : Any]
        playerItemVideoOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: pixelBufferAttributes)

        if let output = playerItemVideoOutput {
            print("Adding AVPlayerItemVideoOutput to player item.")
            item.add(output)
        } else {
            print("Error: Failed to create AVPlayerItemVideoOutput.")
        }
    }

    // Setup the CADisplayLink for frame-synchronized processing
    private func setupDisplayLink() {
        // Invalidate existing link if any
        displayLink?.invalidate()

        // Create a new display link targeting the frame processing method
        displayLink = CADisplayLink(target: self, selector: #selector(displayLinkDidFire))
        displayLink?.add(to: .main, forMode: .common)
        print("CADisplayLink setup.")
    }

    // Called by the CADisplayLink on every screen refresh
    @objc private func displayLinkDidFire(_ link: CADisplayLink) {
        // Ensure player item output exists and player is playing
        guard let output = playerItemVideoOutput, player.timeControlStatus == .playing else { return }

        // Get the current time for the host clock
        let currentTime = CACurrentMediaTime()
        // Ask the output for the item time corresponding to the host time
        let itemTime = output.itemTime(forHostTime: currentTime)

        // Check if there's a new pixel buffer available for this time
        if output.hasNewPixelBuffer(forItemTime: itemTime) {
            // Copy the pixel buffer if available
            if let pixelBuffer = output.copyPixelBuffer(forItemTime: itemTime, itemTimeForDisplay: nil) {
                // Process this frame using Vision, passing the DETECTED orientation
                processFrame(pixelBuffer, orientation: self.videoOrientation)
            }
        }
    }

    // Process a single video frame with Vision to detect body pose
    private func processFrame(_ pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation) {
        do {
            // Perform request with the correct orientation
            try visionSequenceHandler.perform([bodyPoseRequest], on: pixelBuffer, orientation: orientation)

            // Get the results from the request (potentially multiple observations)
            guard let results = bodyPoseRequest.results else {
                // No results, clear poses
                DispatchQueue.main.async {
                    if !self.poses.isEmpty {
                        self.poses = []
                    }
                }
                return
            }

            // Multi-person processing
            var detectedPoses: [[VNHumanBodyPoseObservation.JointName: CGPoint]] = []
            for observation in results {
                let points = extractPoints(from: observation)
                if !points.isEmpty {
                    detectedPoses.append(points)
                }
            }

            // Update the published property on the main thread
            DispatchQueue.main.async {
                self.poses = detectedPoses
            }
        } catch {
            print("Vision performance error: \(error)")
        }
    }

    // Extract joint points from a SINGLE VNHumanBodyPoseObservation
    private func extractPoints(from observation: VNHumanBodyPoseObservation) -> [VNHumanBodyPoseObservation.JointName: CGPoint] {
        var detectedPoints: [VNHumanBodyPoseObservation.JointName: CGPoint] = [:]

        do {
            // Get the recognized points for all available joints in this observation
            let recognizedPoints = try observation.recognizedPoints(.all)

            for (jointName, point) in recognizedPoints {
                // Filter by confidence threshold
                if point.confidence > 0.1 {
                    // Convert to SwiftUI coordinates by flipping Y
                    detectedPoints[jointName] = CGPoint(x: point.location.x, y: 1.0 - point.location.y)
                }
            }
        } catch {
            print("Error getting recognized points for an observation: \(error)")
        }

        return detectedPoints
    }
    
    // Helper function to determine CGImagePropertyOrientation from CGAffineTransform
    private func orientation(from transform: CGAffineTransform) -> CGImagePropertyOrientation {
        let t = transform
        // Analyze the transform matrix to determine orientation
        switch (t.a, t.b, t.c, t.d) {
        case (0.0, 1.0, -1.0, 0.0): // Portrait
            return .right
        case (0.0, -1.0, 1.0, 0.0): // Portrait (Upside Down)
            return .left
        case (-1.0, 0.0, 0.0, -1.0): // Landscape (Left)
            return .down
        case (1.0, 0.0, 0.0, 1.0): // Landscape (Right) - Identity
            return .up
        default: // Default orientation if transform doesn't match known patterns
            print("Warning: Unknown CGAffineTransform found: \(t). Defaulting to .up orientation.")
            return .up
        }
    }

    // MARK: - Playback Control Methods
    func play() {
        if isVideoReady {
            print("Playing video.")
            player.play()
            isPlaying = true
            if displayLink == nil || displayLink?.isPaused == true {
                setupDisplayLink()
            }
            displayLink?.isPaused = false
        } else {
            print("Attempted to play before video was ready.")
        }
    }

    func pause() {
        print("Pausing video.")
        player.pause()
        isPlaying = false
        displayLink?.isPaused = true
    }

    func togglePlayPause() {
        if isPlaying {
            pause()
        } else {
            play()
        }
    }
    
    func restart() {
        print("Restarting video.")
        player.seek(to: .zero) { [weak self] finished in
            if finished {
                self?.play()
            }
        }
    }

    // Clean up resources when the ViewModel is deallocated
    func cleanup() {
        print("Cleaning up VideoPlayerViewModel.")
        // Stop playback
        player.pause()
        isPlaying = false

        // Invalidate display link
        displayLink?.invalidate()
        displayLink = nil

        // Remove KVO observer
        playerItemStatusObserver?.invalidate()
        playerItemStatusObserver = nil

        // Remove NotificationCenter observer
        NotificationCenter.default.removeObserver(self, name: .AVPlayerItemDidPlayToEndTime, object: player.currentItem)
        
        // Cancel Combine subscriptions
        cancellables.forEach { $0.cancel() }
        cancellables.removeAll()

        // Remove video output from player item
        if let output = playerItemVideoOutput, let item = player.currentItem {
            print("Removing AVPlayerItemVideoOutput.")
            item.remove(output)
        }
        playerItemVideoOutput = nil
        
        // Nil out player and item to break potential retain cycles
        player.replaceCurrentItem(with: nil)
    }

    deinit {
        print("VideoPlayerViewModel deinit")
        cleanup()
    }
}
