import SwiftUI
import AVKit

// MARK: - Video with Pose View
struct VideoWithPoseView: View {
    @ObservedObject var viewModel: VideoPlayerViewModel
    @State private var showControls = true
    @State private var controlsTimer: Timer?
    @State private var videoRect: CGRect = .zero

    var body: some View {
        ZStack {
            ColorManager.background.edgesIgnoringSafeArea(.all)

            if viewModel.isVideoReady {
                GeometryReader { geometry in
                    ZStack(alignment: .bottom) {
                        VideoPlayerRepresentable(player: viewModel.player, videoRect: $videoRect)
                            .frame(width: geometry.size.width, height: geometry.size.height)

                        PoseOverlayView(
                            poses: viewModel.poses,
                            connections: viewModel.bodyConnections,
                            videoRect: videoRect
                        )
                        .allowsHitTesting(false)

                        if showControls {
                            playbackControls
                                .padding()
                                .background(ColorManager.background.opacity(0.5))
                                .cornerRadius(10)
                                .padding(.bottom, 30)
                                .transition(.opacity.combined(with: .move(edge: .bottom)))
                                .zIndex(1)
                        }
                    }
                    .contentShape(Rectangle())
                    .onTapGesture {
                        toggleControlsVisibility()
                    }
                }
            } else {
                VStack {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: ColorManager.accentColor))
                        .scaleEffect(1.5)

                    Text(LocalizedStringKey("Preparing video..."))
                        .foregroundColor(ColorManager.textPrimary)
                        .padding(.top, 20)
                }
            }
        }
        .edgesIgnoringSafeArea(.all)
        .onAppear {
            if viewModel.isVideoReady && !viewModel.isPlaying {
                viewModel.play()
                scheduleControlHiding()
            }
        }
        .onDisappear {
            viewModel.pause()
            invalidateControlsTimer()
        }
        .onChange(of: viewModel.isVideoReady) { isReady in
            if isReady && !viewModel.isPlaying {
                viewModel.play()
                scheduleControlHiding()
            }
        }
    }

    private var playbackControls: some View {
        HStack(spacing: 30) {
            Button {
                viewModel.restart()
                resetControlsTimer()
            } label: {
                Image(systemName: "backward.end.fill")
                    .font(.title2)
            }

            Button {
                viewModel.togglePlayPause()
                resetControlsTimer()
            } label: {
                Image(systemName: viewModel.isPlaying ? "pause.fill" : "play.fill")
                    .font(.largeTitle)
            }
            
            Button {} label: {
                Image(systemName: "forward.end.fill")
                    .font(.title2)
                    .opacity(0)
            }
            .disabled(true)
        }
        .foregroundColor(ColorManager.accentColor)
    }
    
    private func toggleControlsVisibility() {
        withAnimation {
            showControls.toggle()
        }
        if showControls {
            scheduleControlHiding()
        } else {
            invalidateControlsTimer()
        }
    }

    private func scheduleControlHiding() {
        invalidateControlsTimer()
        controlsTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: false) { _ in
            DispatchQueue.main.async {
                withAnimation {
                    self.showControls = false
                }
            }
        }
    }

    private func invalidateControlsTimer() {
        controlsTimer?.invalidate()
        controlsTimer = nil
    }

    private func resetControlsTimer() {
        if showControls {
            scheduleControlHiding()
        }
    }
}
