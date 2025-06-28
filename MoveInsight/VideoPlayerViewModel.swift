// MoveInsight/VideoPlayerViewModel.swift
import SwiftUI
import AVKit
import Combine
import simd // For SIMD3<Float>

// MARK: - Video Player View Model
class VideoPlayerViewModel: ObservableObject {
    let videoURL: URL
    let player: AVPlayer
    let asset: AVAsset

    // Published properties to update the UI
    // Poses will now be [String: SIMD3<Float>] to store 3D coordinates from the server.
    @Published var poses: [String: SIMD3<Float>] = [:]
    // accumulatedPoses stores all frames' 3D poses after processing from server.
    @Published var accumulatedPoses: [[String: SIMD3<Float>]] = []
    
    // Store the original server frames for potential debugging or re-processing.
    @Published var originalServerFrames: [ServerFrameData] = []

    // bodyConnections are used for drawing the 2D overlay skeleton.
    @Published var bodyConnections: [StringBodyConnection] = []
    
    @Published var isVideoReady = false
    @Published var isPlaying = false
    @Published var videoOrientation: CGImagePropertyOrientation = .up // For potential orientation adjustments
    
    private(set) var videoSize: CGSize = .zero // Actual video dimensions
    let videoSource: Pose3DBody.VideoSource // To distinguish between user and model videos
    
    private var displayLink: CADisplayLink?
    private var playerItemStatusObserver: NSKeyValueObservation?
    private var didPlayToEndTimeObserver: NSObjectProtocol?
    private var cancellables = Set<AnyCancellable>()

    init(videoURL: URL, videoSource: Pose3DBody.VideoSource) {
        self.videoURL = videoURL
        self.asset = AVAsset(url: videoURL)
        self.player = AVPlayer() // Initialize player
        self.videoSource = videoSource
        print("VideoPlayerViewModel (3D enabled) initialized with URL: \(videoURL.lastPathComponent) for source: \(videoSource)")
        setupStringBodyConnectionsFor2DOverlay() // Setup connections for 2D overlay
        prepareVideoPlayback() // Asynchronously prepare video
    }
    
    // Called to populate the ViewModel with 3D pose data from the server.
    func setServerProcessedJoints(_ serverFrames: [ServerFrameData]) {
        self.originalServerFrames = serverFrames // Store the raw server response

        var allFramesPoses3D: [[String: SIMD3<Float>]] = []
        for serverFrameData in serverFrames {
            var singleFramePoses3D: [String: SIMD3<Float>] = [:]
            for (jointNameString, jointData) in serverFrameData.joints {
                // Convert server's x, y, z (Double?) to SIMD3<Float>.
                // Server's x,y are normalized (0-1). Z's scale is relative to MediaPipe's world coordinates.
                // The client-side SceneView3D will handle appropriate scaling of Z for rendering.
                let x = Float(jointData.x)
                let y = Float(jointData.y)
                let z = Float(jointData.z ?? 0.0) // Default z to 0.0 if server sends nil
                singleFramePoses3D[jointNameString] = SIMD3<Float>(x, y, z)
            }
            if !singleFramePoses3D.isEmpty { // Only add if there are joints for this frame
                allFramesPoses3D.append(singleFramePoses3D)
            }
        }
        
        DispatchQueue.main.async {
            self.accumulatedPoses = allFramesPoses3D
            print("Successfully set \(self.accumulatedPoses.count) frames of 3D joint data for source: \(self.videoSource).")
            // If video is at the beginning and poses are loaded, set the first frame's pose.
            if !self.accumulatedPoses.isEmpty && self.player.currentTime() == .zero {
                 self.poses = self.accumulatedPoses[0]
            }
        }
    }

    // Defines the connections for the 2D skeleton overlay.
    private func setupStringBodyConnectionsFor2DOverlay() {
        // These connections use standard MediaPipe joint names (or remapped ones like "LeftToe").
        // Ensure these names match what the server provides and what PoseOverlayView expects.
        bodyConnections = [
            // Torso
            StringBodyConnection(from: "Nose", to: "LeftShoulder"),
            StringBodyConnection(from: "Nose", to: "RightShoulder"),
            StringBodyConnection(from: "LeftShoulder", to: "RightShoulder"),
            StringBodyConnection(from: "LeftShoulder", to: "LeftHip"),
            StringBodyConnection(from: "RightShoulder", to: "RightHip"),
            StringBodyConnection(from: "LeftHip", to: "RightHip"),

            // Left Arm
            StringBodyConnection(from: "LeftShoulder", to: "LeftElbow"),
            StringBodyConnection(from: "LeftElbow", to: "LeftWrist"),
            
            // Right Arm
            StringBodyConnection(from: "RightShoulder", to: "RightElbow"),
            StringBodyConnection(from: "RightElbow", to: "RightWrist"),

            // Left Leg
            StringBodyConnection(from: "LeftHip", to: "LeftKnee"),
            StringBodyConnection(from: "LeftKnee", to: "LeftAnkle"),
            StringBodyConnection(from: "LeftAnkle", to: "LeftHeel"),
            StringBodyConnection(from: "LeftHeel", to: "LeftToe"), // "LeftToe" is often "LeftFootIndex" from MediaPipe

            // Right Leg
            StringBodyConnection(from: "RightHip", to: "RightKnee"),
            StringBodyConnection(from: "RightKnee", to: "RightAnkle"),
            StringBodyConnection(from: "RightAnkle", to: "RightHeel"),
            StringBodyConnection(from: "RightHeel", to: "RightToe") // "RightToe" is often "RightFootIndex"
        ]
    }

    // Asynchronously loads video properties and sets up the player item.
    private func prepareVideoPlayback() {
        Task { [weak self] in // Use weak self to avoid retain cycles
            guard let self = self else { return }

            do {
                // Load video tracks to get orientation and size
                let tracks = try await self.asset.load(.tracks)
                if let videoTrack = tracks.first(where: { $0.mediaType == .video }) {
                    let transform = try await videoTrack.load(.preferredTransform)
                    self.videoOrientation = self.orientation(from: transform) // Determine video orientation
                    self.videoSize = try await videoTrack.load(.naturalSize) // Get natural video size
                    
                    // Switch to main thread to update UI-related properties and setup player
                    await MainActor.run {
                        self.setupPlayerItemAndObservers()
                    }
                } else {
                    print("Error: No video track found in asset for \(self.videoSource) - URL: \(self.videoURL.lastPathComponent).")
                    await MainActor.run { self.isVideoReady = false }
                }
            } catch {
                print("Error loading asset tracks or properties for \(self.videoSource) - URL: \(self.videoURL.lastPathComponent): \(error)")
                await MainActor.run { self.isVideoReady = false }
            }
        }
    }
    
    // Sets up the AVPlayerItem and observers for playback status.
    private func setupPlayerItemAndObservers() {
        let playerItem = AVPlayerItem(asset: asset)
        
        // Observe player item status to know when it's ready to play
        playerItemStatusObserver = playerItem.observe(\.status, options: [.new, .initial]) { [weak self] item, _ in
            guard let self = self else { return }
            DispatchQueue.main.async {
                switch item.status {
                case .readyToPlay:
                    self.isVideoReady = true
                    self.setupDisplayLink() // Start display link for frame-by-frame updates
                    print("PlayerItem ready for \(self.videoSource) (3D). Video size: \(self.videoSize). URL: \(self.videoURL.lastPathComponent)")
                    // If poses are already loaded and video is at start, display the first pose
                    if !self.accumulatedPoses.isEmpty && self.player.currentTime().seconds == 0 {
                        self.poses = self.accumulatedPoses[0]
                    }
                case .failed:
                    let errorDesc = item.error?.localizedDescription ?? "Unknown error"
                    print("PlayerItem failed for \(self.videoSource) (URL: \(self.videoURL.lastPathComponent)): \(errorDesc).")
                    self.isVideoReady = false
                default: // .unknown
                    self.isVideoReady = false
                }
            }
        }

        // Observe when video plays to the end
        didPlayToEndTimeObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: playerItem,
            queue: .main
        ) { [weak self] _ in
            self?.isPlaying = false
            // Optionally, seek to zero to allow replay:
            // self?.player.seek(to: .zero)
        }
        
        player.replaceCurrentItem(with: playerItem)
        player.isMuted = (videoSource == .secondary) // Mute model/secondary video by default
        player.allowsExternalPlayback = true
    }

    // Sets up the CADisplayLink for synchronizing pose updates with screen refresh rate.
    private func setupDisplayLink() {
        displayLink?.invalidate() // Invalidate existing display link if any
        displayLink = CADisplayLink(target: self, selector: #selector(displayLinkDidFire))
        displayLink?.add(to: .main, forMode: .common) // Add to main run loop
        displayLink?.isPaused = !isPlaying // Pause if not playing
    }

    // Called by CADisplayLink on each screen refresh.
    @objc private func displayLinkDidFire(_ link: CADisplayLink) {
        // Only update poses if player is playing and pose data is available.
        guard player.timeControlStatus == .playing, !accumulatedPoses.isEmpty else {
            return
        }
        
        let currentTime = player.currentTime()
        // Ensure video track and frame rate are available for accurate frame indexing.
        guard let videoTrack = asset.tracks(withMediaType: .video).first,
              videoTrack.nominalFrameRate > 0 else {
            // If no track info, clear current poses to avoid displaying stale data.
            if !poses.isEmpty { DispatchQueue.main.async { self.poses = [:] } }
            return
        }
        
        let frameRate = videoTrack.nominalFrameRate
        let currentFrameIndex = Int(CMTimeGetSeconds(currentTime) * Double(frameRate))

        // Update current poses if the frame index is valid.
        if currentFrameIndex >= 0 && currentFrameIndex < accumulatedPoses.count {
            let newPoseForCurrentFrame = accumulatedPoses[currentFrameIndex]
            // Only publish update if the pose has actually changed to prevent redundant UI updates.
            if self.poses != newPoseForCurrentFrame {
                 DispatchQueue.main.async { // Ensure UI updates are on the main thread
                    self.poses = newPoseForCurrentFrame
                 }
            }
        } else {
            // Frame index is out of bounds (e.g., video ended or data issue). Clear current poses.
            if !poses.isEmpty { DispatchQueue.main.async { self.poses = [:] } }
        }
    }
    
    // Determines video orientation from its transform.
    private func orientation(from transform: CGAffineTransform) -> CGImagePropertyOrientation {
        if transform.a == 0 && transform.b == 1.0 && transform.c == -1.0 && transform.d == 0 { return .right }
        if transform.a == 0 && transform.b == -1.0 && transform.c == 1.0 && transform.d == 0 { return .left }
        if transform.a == -1.0 && transform.b == 0 && transform.c == 0 && transform.d == -1.0 { return .down }
        return .up // Default orientation
    }

    // MARK: - Playback Controls
    func play() {
        if isVideoReady && !isPlaying {
            player.play()
            isPlaying = true
            displayLink?.isPaused = false // Resume display link
        }
    }

    func pause() {
        if isPlaying {
            player.pause()
            isPlaying = false
            displayLink?.isPaused = true // Pause display link
            // Trigger one last pose update on pause to ensure the correct frame's pose is displayed.
            if let currentDisplayLink = displayLink {
                displayLinkDidFire(currentDisplayLink)
            }
        }
    }

    func togglePlayPause() {
        if isPlaying { pause() } else { play() }
    }
    
    func restart() {
        player.seek(to: .zero) { [weak self] finished in
            guard let self = self else { return }
            if finished {
                // Immediately update to the first frame's pose after seeking.
                if !self.accumulatedPoses.isEmpty {
                    DispatchQueue.main.async {
                        self.poses = self.accumulatedPoses[0]
                    }
                }
                self.play() // Start playing after seeking to beginning
            }
        }
    }
    
    // Clears all loaded pose data.
    func clearAllPoseData() {
        DispatchQueue.main.async {
            self.accumulatedPoses.removeAll()
            self.poses.removeAll()
            self.originalServerFrames.removeAll()
            print("Cleared all 3D pose data for source: \(self.videoSource). URL: \(self.videoURL.lastPathComponent)")
        }
    }

    // MARK: - Cleanup
    // Call this method when the ViewModel is no longer needed to release resources.
    func cleanup() {
        print("VideoPlayerViewModel (3D) cleanup initiated for \(videoSource). URL: \(self.videoURL.lastPathComponent)")
        pause() // Ensure player and display link are paused
        player.replaceCurrentItem(with: nil) // Release current player item
        
        displayLink?.invalidate() // Stop and remove display link
        displayLink = nil
        
        playerItemStatusObserver?.invalidate() // Remove KVO
        playerItemStatusObserver = nil
        
        if let observer = didPlayToEndTimeObserver { // Remove notification observer
            NotificationCenter.default.removeObserver(observer)
            didPlayToEndTimeObserver = nil
        }
        
        cancellables.forEach { $0.cancel() } // Cancel any Combine subscriptions
        cancellables.removeAll()
        
        clearAllPoseData() // Clear pose data arrays
        print("VideoPlayerViewModel (3D) cleaned up for \(videoSource). URL: \(self.videoURL.lastPathComponent)")
    }

    deinit {
        print("VideoPlayerViewModel (3D) deinit for \(videoSource). URL: \(self.videoURL.lastPathComponent)")
        // Cleanup should ideally be called explicitly by the owner of this ViewModel.
        // However, as a safeguard, attempt cleanup here too.
        cleanup()
    }
}

// Ensure Pose3DBody.VideoSource is defined, e.g., in BodyPoseTypes.swift
// enum VideoSource { case primary, secondary }
// struct Pose3DBody { let videoSource: VideoSource; /* ... other properties ... */ }
