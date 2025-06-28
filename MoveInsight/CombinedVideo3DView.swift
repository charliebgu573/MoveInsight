// MoveInsight/CombinedVideo3DView.swift
import SwiftUI
import AVKit
import Combine
import simd // Required for SIMD3 if PoseOverlayViewForComparison uses it via ViewModel

// MARK: - Pose Overlay View for Comparison (Adapted for 3D Poses, Renders 2D Projection)
// This view is used by CombinedVideo2DComparisonView to display 2D projections of 3D poses.
struct PoseOverlayViewForComparison: View {
    @ObservedObject var viewModel: VideoPlayerViewModel // viewModel.poses is now [String: SIMD3<Float>]
    let videoActualRect: CGRect // The actual rectangle of the video content within its player view

    var body: some View {
        Canvas { context, size in
            // Ensure poses are available and the video rectangle is valid.
            guard !viewModel.poses.isEmpty,
                  videoActualRect.size.width > 0,
                  videoActualRect.size.height > 0 else {
                // print("PoseOverlayViewForComparison: No poses or invalid videoActualRect \(videoActualRect)")
                return
            }
            let currentPoses3D = viewModel.poses // These are [String: SIMD3<Float>]

            // Draw connections using the x and y components of the 3D poses.
            for connection in viewModel.bodyConnections { // bodyConnections are StringBodyConnection for 2D
                guard let fromPose3D = currentPoses3D[connection.from],
                      let toPose3D = currentPoses3D[connection.to] else {
                    // print("PoseOverlayViewForComparison: Missing joint for connection \(connection.from) -> \(connection.to)")
                    continue
                }
                
                // Extract 2D points (normalized x, y) from the 3D pose data.
                let fromPointNorm = CGPoint(x: CGFloat(fromPose3D.x), y: CGFloat(fromPose3D.y))
                let toPointNorm = CGPoint(x: CGFloat(toPose3D.x), y: CGFloat(toPose3D.y))
                
                // Scale these normalized points to the actual video rectangle's coordinate system.
                let fromCanvasPoint = CGPoint(
                    x: videoActualRect.origin.x + (fromPointNorm.x * videoActualRect.size.width),
                    y: videoActualRect.origin.y + (fromPointNorm.y * videoActualRect.size.height)
                )
                let toCanvasPoint = CGPoint(
                    x: videoActualRect.origin.x + (toPointNorm.x * videoActualRect.size.width),
                    y: videoActualRect.origin.y + (toPointNorm.y * videoActualRect.size.height)
                )
                
                var path = Path()
                path.move(to: fromCanvasPoint)
                path.addLine(to: toCanvasPoint)
                context.stroke(path, with: .color(ColorManager.accentColor.opacity(0.8)), lineWidth: 3)
            }

            // Draw joints using the x and y components.
            for (_, jointPose3D) in currentPoses3D {
                let jointPointNorm = CGPoint(x: CGFloat(jointPose3D.x), y: CGFloat(jointPose3D.y))
                let jointCanvasPoint = CGPoint(
                    x: videoActualRect.origin.x + (jointPointNorm.x * videoActualRect.size.width),
                    y: videoActualRect.origin.y + (jointPointNorm.y * videoActualRect.size.height)
                )
                let rect = CGRect(x: jointCanvasPoint.x - 4, y: jointCanvasPoint.y - 4, width: 8, height: 8)
                context.fill(Path(ellipseIn: rect), with: .color(Color.red.opacity(0.8)))
            }
        }
        .opacity(0.7)
    }
}


// MARK: - Combined 2D Video Comparison View
// This view displays two video players side-by-side, each with a 2D pose overlay.
// It's used when a simple 2D side-by-side comparison is desired, even with 3D data.
// The "3D Overlay" mode in TechniqueComparisonView will use SceneView3D directly.
struct CombinedVideo2DComparisonView: View {
    @ObservedObject var primaryViewModel: VideoPlayerViewModel   // ViewModel now handles 3D poses
    @ObservedObject var secondaryViewModel: VideoPlayerViewModel // ViewModel now handles 3D poses
    
    // State for synchronized playback control
    @State private var currentTime: Double = 0
    @State private var isPlayingUserInitiated: Bool = false // Tracks if user explicitly hit play
    @State private var duration: Double = 0.1 // Default duration to prevent division by zero
    @State private var isLoadingDuration: Bool = true // To show loading state for duration

    // State variables to hold the actual video rectangles for each player's 2D overlay.
    @State private var primaryVideoActualRect: CGRect = .zero
    @State private var secondaryVideoActualRect: CGRect = .zero

    @State private var timeObserverToken: Any? = nil
    private let frameStep: Double = 1.0 / 30.0 // Assuming 30fps for step controls

    var body: some View {
        ZStack {
            ColorManager.background.ignoresSafeArea() // Ensure background color is applied

            VStack(spacing: 0) { // Use 0 spacing for tighter layout if desired
                if primaryViewModel.isVideoReady && secondaryViewModel.isVideoReady && !isLoadingDuration {
                    HStack(spacing: 8) { // Spacing between the two video players
                        // Primary video player with its overlay
                        videoPlayerWithOverlay(
                            for: primaryViewModel,
                            title: "Your Video", // Or LocalizedStringKey("Your Technique")
                            videoActualRectBinding: $primaryVideoActualRect
                        )
                        // Secondary video player with its overlay
                        videoPlayerWithOverlay(
                            for: secondaryViewModel,
                            title: "Comparison Video", // Or LocalizedStringKey("Model Technique")
                            videoActualRectBinding: $secondaryVideoActualRect
                        )
                    }
                    .padding(.horizontal, 8) // Padding around the HStack of players
                    .padding(.top, 8)
                    
                    controlBar // Playback controls
                        .padding(.vertical, 10)

                } else {
                    // Loading indicator while videos or durations are loading
                    ProgressView(isLoadingDuration ? "Loading video durations..." : "Loading Videos for Comparison...")
                        .tint(ColorManager.accentColor)
                        .frame(maxWidth: .infinity, maxHeight: .infinity) // Center the progress view
                }
            }
        }
        .onAppear {
            isLoadingDuration = true
            // Asynchronously load video durations
            Task {
                do {
                    let primaryDurationSeconds = try await primaryViewModel.asset.load(.duration).seconds
                    let secondaryDurationSeconds = try await secondaryViewModel.asset.load(.duration).seconds
                    await MainActor.run {
                        self.duration = max(primaryDurationSeconds, secondaryDurationSeconds, 0.1) // Ensure duration is at least 0.1
                        self.isLoadingDuration = false
                    }
                } catch {
                    await MainActor.run {
                        print("Error loading video durations for CombinedVideo2DComparisonView: \(error)")
                        self.duration = 0.1 // Fallback duration
                        self.isLoadingDuration = false
                    }
                }
            }
            
            setupTimeObserver() // Setup observer for synchronized playback
            primaryViewModel.player.isMuted = false // User's video is unmuted
            secondaryViewModel.player.isMuted = true // Comparison video is muted
            
            // If either video was already playing (e.g., from a previous view state)
            if primaryViewModel.isPlaying || secondaryViewModel.isPlaying {
                isPlayingUserInitiated = true // Reflect that playback was intended
                primaryViewModel.play()
                secondaryViewModel.play()
            }
        }
        .onDisappear {
            removeTimeObserver() // Clean up time observer
            primaryViewModel.pause() // Pause videos when view disappears
            secondaryViewModel.pause()
        }
    }
    
    // Helper view for a single video player card with its 2D overlay.
    @ViewBuilder
    private func videoPlayerWithOverlay(
        for viewModel: VideoPlayerViewModel,
        title: String, // Or LocalizedStringKey
        videoActualRectBinding: Binding<CGRect> // Binding for this player's video rect
    ) -> some View {
        VStack {
            Text(title)
                .font(.caption)
                .foregroundColor(ColorManager.textSecondary)
            
            // VideoPlayerRepresentable displays the video.
            // PoseOverlayViewForComparison displays the 2D projection of 3D poses.
            VideoPlayerRepresentable(player: viewModel.player, videoRect: videoActualRectBinding)
                .overlay(PoseOverlayViewForComparison(viewModel: viewModel, videoActualRect: videoActualRectBinding.wrappedValue))
                .aspectRatio(CGSize(width: 9, height: 16), contentMode: .fit) // Maintain aspect ratio
                .background(Color.black) // Black background for letter/pillarboxing
                .cornerRadius(8)
                .shadow(radius: 3)
        }
    }
    
    // Playback control bar (Slider, Play/Pause, Step buttons).
    // Implementation remains the same as it controls AVPlayer and UI state, not pose data directly.
    private var controlBar: some View { /* ... as before ... */
        HStack(spacing: 15) {
            Button(action: togglePlayPause) {
                Image(systemName: isPlayingUserInitiated && primaryViewModel.player.rate != 0 ? "pause.fill" : "play.fill")
                    .font(.title2)
            }

            Button(action: { step(by: -1) }) { Image(systemName: "backward.frame.fill").font(.title3) }
            
            Slider( value: $currentTime, in: 0...duration, onEditingChanged: sliderEditingChanged )
                .accentColor(ColorManager.accentColor)
                .disabled(duration <= 0.1 || isLoadingDuration)

            Button(action: { step(by: 1) }) { Image(systemName: "forward.frame.fill").font(.title3) }
            
            Text(String(format: "%.2f / %.2f", currentTime, duration))
                .font(.caption).foregroundColor(ColorManager.textSecondary).frame(minWidth: 80, alignment: .leading)
        }
        .padding(.horizontal).foregroundColor(ColorManager.accentColor)
    }

    private func sliderEditingChanged(editing: Bool) {
        if editing { // User started scrubbing
            if primaryViewModel.player.rate != 0 { primaryViewModel.pause() }
            if secondaryViewModel.player.rate != 0 { secondaryViewModel.pause() }
        } else { // User finished scrubbing
            seek(to: currentTime)
            if isPlayingUserInitiated { // Resume playback if it was playing before scrubbing
                primaryViewModel.play()
                secondaryViewModel.play()
            }
        }
    }

    private func togglePlayPause() { /* ... as before ... */
        isPlayingUserInitiated.toggle()
        if isPlayingUserInitiated { primaryViewModel.play(); secondaryViewModel.play() }
        else { primaryViewModel.pause(); secondaryViewModel.pause() }
    }

    private func step(by frames: Int) { /* ... as before ... */
        guard duration > 0.1 && !isLoadingDuration else { return }
        let newTime = max(0, min(duration, currentTime + Double(frames) * frameStep))
        seek(to: newTime) // Seek to the new time
        
        // If paused, briefly play/pause to update the visual frame for both players
        if !isPlayingUserInitiated {
            primaryViewModel.play(); secondaryViewModel.play()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) { // Short delay
                primaryViewModel.pause(); secondaryViewModel.pause()
                // Explicitly update currentTime here if slider doesn't update automatically when paused
                // self.currentTime = newTime
            }
        }
        // If already playing, seek will update the frame, and currentTime updates via timeObserver.
    }

    private func seek(to timeSec: Double) { /* ... as before ... */
        guard duration > 0.1 && !isLoadingDuration else { return }
        let cmTime = CMTime(seconds: timeSec, preferredTimescale: 600)
        let dispatchGroup = DispatchGroup()
        let wasPlaying = isPlayingUserInitiated && primaryViewModel.player.rate != 0

        // Pause before seeking for smoother experience
        if primaryViewModel.player.rate != 0 { primaryViewModel.pause() }
        if secondaryViewModel.player.rate != 0 { secondaryViewModel.pause() }

        dispatchGroup.enter()
        primaryViewModel.player.seek(to: cmTime, toleranceBefore: .zero, toleranceAfter: .zero) { _ in dispatchGroup.leave() }
        
        dispatchGroup.enter()
        secondaryViewModel.player.seek(to: cmTime, toleranceBefore: .zero, toleranceAfter: .zero) { _ in dispatchGroup.leave() }

        dispatchGroup.notify(queue: .main) {
            self.currentTime = timeSec // Ensure slider reflects the seeked time
            if wasPlaying { // Resume playback if it was playing before seek
                primaryViewModel.play(); secondaryViewModel.play()
            }
            // If paused, the VideoPlayerViewModel's displayLinkDidFire (if manually triggered on pause)
            // or the next play action would update the `poses` for the current frame.
        }
    }

    private func setupTimeObserver() { /* ... as before ... */
        guard timeObserverToken == nil else { return }
        // More frequent updates for smoother slider sync, e.g., half of frameStep
        let interval = CMTime(seconds: frameStep / 2.0, preferredTimescale: CMTimeScale(NSEC_PER_SEC))
        
        timeObserverToken = primaryViewModel.player.addPeriodicTimeObserver(forInterval: interval, queue: .main) { [self] time in
            if !duration.isZero && !isLoadingDuration {
                 self.currentTime = time.seconds
            }
        }
    }
    private func removeTimeObserver() { /* ... as before ... */
        if let token = timeObserverToken {
            primaryViewModel.player.removeTimeObserver(token)
            timeObserverToken = nil
        }
    }
}
