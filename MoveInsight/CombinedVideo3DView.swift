import SwiftUI
import AVKit
import Vision
import simd

struct CombinedVideo3DView: View {
    @ObservedObject var baseViewModel: VideoPlayerViewModel
    @ObservedObject var overlayViewModel: VideoPlayerViewModel
    
    // Playback control state
    @State private var currentTime: Double = 0
    @State private var timeObserver: Any? = nil
    private var duration: Double { baseViewModel.asset.duration.seconds }
    private let frameStep: Double = 1.0 / 30.0

    var body: some View {
        ZStack {
            ColorManager.background.ignoresSafeArea()

            VStack(spacing: 0) {
                ZStack {
                    ColorManager.background.ignoresSafeArea()

                    if baseViewModel.isVideoReady && overlayViewModel.isVideoReady {
                        // Just show the SceneView - no toggles or previews
                        SceneView3D(
                            pose3DBodies: combinedPose3DBodies,
                            bodyConnections: baseViewModel.bodyConnections
                        )
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else {
                        ProgressView("Loading 3D view...")
                            .tint(ColorManager.accentColor)
                    }
                }
                .onAppear {
                    setupTimeObserver()
                    // Start both videos: base audible, overlay muted
                    baseViewModel.play()
                    overlayViewModel.play()
                    overlayViewModel.player.isMuted = true
                }
                .onDisappear {
                    removeTimeObserver()
                    baseViewModel.pause()
                    overlayViewModel.pause()
                }

                // Shared controls for both videos
                controlBar
            }
        }
    }
    
    private var combinedPose3DBodies: [Pose3DBody] {
        baseViewModel.pose3DBodies + overlayViewModel.pose3DBodies
    }

    private var controlBar: some View {
        HStack(spacing: 20) {
            Button(action: togglePlayPause) {
                Image(systemName: baseViewModel.isPlaying ? "pause.fill" : "play.fill")
            }

            Button(action: { step(by: -1) }) {
                Image(systemName: "backward.frame")
            }

            Button(action: { step(by: 1) }) {
                Image(systemName: "forward.frame")
            }

            Slider(
                value: $currentTime,
                in: 0...max(0.1, duration), // Ensure non-zero range
                onEditingChanged: { editing in
                    if !editing {
                        seek(to: currentTime)
                    }
                }
            )
        }
        .padding()
        .background(ColorManager.background.opacity(0.7))
    }

    // MARK: - Playback helpers controlling both videos
    private func togglePlayPause() {
        if baseViewModel.isPlaying {
            baseViewModel.pause()
            overlayViewModel.pause()
        } else {
            baseViewModel.play()
            overlayViewModel.play()
        }
    }

    private func step(by frames: Int) {
        let newTime = max(0, min(duration, currentTime + Double(frames) * frameStep))
        seek(to: newTime)
    }

    private func seek(to timeSec: Double) {
        let cmTime = CMTime(seconds: timeSec, preferredTimescale: 600)

        // Pause both players
        baseViewModel.pause()
        overlayViewModel.pause()

        // Seek both players exactly
        baseViewModel.player.seek(to: cmTime, toleranceBefore: .zero, toleranceAfter: .zero)
        overlayViewModel.player.seek(to: cmTime, toleranceBefore: .zero, toleranceAfter: .zero)

        // Update state
        baseViewModel.isPlaying = false
        overlayViewModel.isPlaying = false
        currentTime = timeSec

        // Briefly unpause to update poses, then pause again
        baseViewModel.play()
        overlayViewModel.play()
        DispatchQueue.main.asyncAfter(deadline: .now() + frameStep) {
            baseViewModel.pause()
            overlayViewModel.pause()
        }
    }

    private func setupTimeObserver() {
        guard timeObserver == nil else { return }
        let interval = CMTime(seconds: frameStep, preferredTimescale: 600)
        let obs = baseViewModel.player.addPeriodicTimeObserver(
            forInterval: interval, queue: .main
        ) { time in
            currentTime = time.seconds
        }
        timeObserver = obs
    }

    private func removeTimeObserver() {
        if let obs = timeObserver {
            baseViewModel.player.removeTimeObserver(obs)
            timeObserver = nil
        }
    }
}
