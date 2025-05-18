// MoveInsight/TechniqueDetailView.swift
import SwiftUI
import AVKit
import Combine
import simd // Required for SIMD3 if used by PoseOverlayView indirectly via ViewModel

// MARK: - Pose Overlay View (Adapted for 3D Poses from ViewModel, Renders 2D Projection)
// This view displays a 2D projection of the 3D pose data onto the video.
struct PoseOverlayView: View {
    @ObservedObject var viewModel: VideoPlayerViewModel // viewModel.poses is now [String: SIMD3<Float>]
    let videoActualRect: CGRect // The actual rectangle of the video content within the player view

    var body: some View {
        Canvas { context, size in
            // Ensure poses are available and the video rectangle is valid.
            guard !viewModel.poses.isEmpty,
                  videoActualRect.size.width > 0,
                  videoActualRect.size.height > 0 else {
                // print("PoseOverlayView: No poses or invalid videoActualRect \(videoActualRect)")
                return
            }
            let currentPoses3D = viewModel.poses // These are [String: SIMD3<Float>]

            // Draw connections using the x and y components of the 3D poses.
            for connection in viewModel.bodyConnections { // bodyConnections are StringBodyConnection for 2D
                guard let fromPose3D = currentPoses3D[connection.from],
                      let toPose3D = currentPoses3D[connection.to] else {
                    // print("PoseOverlayView: Missing joint for connection \(connection.from) -> \(connection.to)")
                    continue
                }
                
                // Extract 2D points (normalized x, y) from the 3D pose data.
                // Server provides x,y as normalized (0-1) screen coordinates.
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

            // Draw joints using the x and y components of the 3D poses.
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
        .opacity(0.7) // Keep overlay slightly transparent.
        // .drawingGroup() // Consider for performance with very complex drawings, test if needed.
    }
}

// KeyPointRow struct remains unchanged.
struct KeyPointRow: View {
    let icon: String
    let text: String
    var body: some View { /* ... as before ... */
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 18))
                .foregroundColor(ColorManager.accentColor)
                .frame(width: 24, height: 24)
            Text(LocalizedStringKey(text))
                .font(.subheadline)
                .foregroundColor(ColorManager.textPrimary)
            Spacer()
        }
    }
}

// MARK: - Main Detail View (TechniqueDetailView)
struct TechniqueDetailView: View {
    let technique: BadmintonTechnique
    
    // StateObjects to manage VideoPlayerViewModels for primary and comparison videos.
    // VideoPlayerViewModel now handles 3D pose data internally.
    @StateObject private var videoVMContainer = VideoVMContainer()
    @StateObject private var comparisonVideoVMContainer = VideoVMContainer()
    
    // UI state variables
    @State private var showUploadOptions = false
    @State private var isVideoBeingUploadedOrProcessed = false
    @State private var processingStatusMessage = NSLocalizedString("Processing video...", comment: "")
    @State private var analysisError: String? = nil
    
    @State private var isUploadingForComparison = false
    @State private var navigateToComparisonView = false // For programmatic navigation
    
    @State private var shouldShowProcessedPrimaryVideo = false

    // State to hold server frame data (which now includes 3D joint info).
    @State private var primaryVideoServerFrames: [ServerFrameData]? = nil
    @State private var comparisonVideoServerFrames: [ServerFrameData]? = nil
    // ComparisonResult is still based on 2D swing analysis from the server.
    @State private var comparisonAnalysisResult: ComparisonResult? = nil
    @State private var isFetchingComparisonAnalysis = false

    // State to hold the actual video rectangle for the primary video player's 2D overlay.
    @State private var primaryVideoActualRect: CGRect = .zero

    @State private var cancellables = Set<AnyCancellable>()
    private let analysisService = TechniqueAnalysisService() // Service handles 3D data communication.
    
    var body: some View {
        ZStack {
            ColorManager.background.ignoresSafeArea()
            
            ScrollView {
                VStack(spacing: 24) {
                    techniqueHeader // Displays technique name, icon, description.
                    
                    // Conditional content based on processing state:
                    if isVideoBeingUploadedOrProcessed {
                        processingIndicator // Shows progress view and status message.
                    } else if isFetchingComparisonAnalysis {
                        // Indicator for when comparison analysis is being fetched.
                        VStack(spacing: 16) {
                            ProgressView().progressViewStyle(CircularProgressViewStyle(tint: ColorManager.accentColor)).scaleEffect(1.5)
                            Text(LocalizedStringKey("Comparing techniques..."))
                                .foregroundColor(ColorManager.textPrimary)
                        }
                        .frame(height: 300).frame(maxWidth: .infinity).background(Color.black.opacity(0.05)).cornerRadius(12).padding(.horizontal)
                    } else if let error = analysisError {
                        errorView(error) // Displays error message and retry option.
                    } else if shouldShowProcessedPrimaryVideo, let primaryVM = videoVMContainer.viewModel {
                        // Display processed primary video with 2D overlay and comparison options.
                        processedVideoView(viewModel: primaryVM, videoActualRect: $primaryVideoActualRect)
                        comparisonOptionsView(primaryVM: primaryVM)
                    } else {
                        // Initial state: show upload button for primary video.
                        uploadButton(isComparisonUpload: false)
                    }
                    
                    keyPointsSection // Displays general key points for the technique.
                }
                .padding(.bottom, 32)
            }
            // Hidden NavigationLink for programmatic navigation to TechniqueComparisonView.
            .background(
                NavigationLink(
                    destination: navigationDestinationView(),
                    isActive: $navigateToComparisonView
                ) { EmptyView() }
            )
        }
        .navigationTitle(technique.name)
        .navigationBarTitleDisplayMode(.inline)
        // Sheet for presenting the video upload view.
        .sheet(isPresented: $showUploadOptions) {
            TechniqueVideoUploadView(technique: technique, isComparison: isUploadingForComparison) { videoURL in
                self.showUploadOptions = false // Dismiss sheet.
                if let url = videoURL { // If a video URL was selected.
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { // Slight delay for UI.
                        if self.isUploadingForComparison {
                            handleComparisonVideoSelected(url)
                        } else {
                            handlePrimaryVideoSelected(url)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Subviews (Header, Indicators, Error, Upload Button, Processed Video, Options, Key Points)
    // These subviews' implementations are largely the same as before, focusing on UI presentation.
    // The `processedVideoView` is notable as it uses `PoseOverlayView` which now handles 3D data for 2D projection.

    private var techniqueHeader: some View { /* ... as before ... */
        VStack(spacing: 12) {
            ZStack {
                Circle().fill(ColorManager.accentColor.opacity(0.2)).frame(width: 80, height: 80)
                Image(systemName: technique.iconName).font(.system(size: 36)).foregroundColor(ColorManager.accentColor)
            }
            Text(technique.name).font(.title).foregroundColor(ColorManager.textPrimary).multilineTextAlignment(.center)
            Text(technique.description).font(.body).foregroundColor(ColorManager.textSecondary).multilineTextAlignment(.center).padding(.horizontal, 24)
        }.padding(.top, 24)
    }

    private var processingIndicator: some View { /* ... as before ... */
        VStack(spacing: 16) {
            ProgressView().progressViewStyle(CircularProgressViewStyle(tint: ColorManager.accentColor)).scaleEffect(1.5)
            Text(LocalizedStringKey(processingStatusMessage))
                .foregroundColor(ColorManager.textPrimary)
        }
        .frame(height: 300).frame(maxWidth: .infinity).background(Color.black.opacity(0.05)).cornerRadius(12).padding(.horizontal)
    }

    private func errorView(_ errorMessage: String) -> some View { /* ... as before ... */
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle.fill").font(.largeTitle).foregroundColor(.red)
            Text(LocalizedStringKey("Error")).font(.title2).foregroundColor(ColorManager.textPrimary)
            Text(errorMessage).font(.body).foregroundColor(ColorManager.textSecondary).multilineTextAlignment(.center).padding(.horizontal)
            Button(LocalizedStringKey("Try Upload Again")) {
                resetToInitialUploadState()
                showUploadOptions = true
            }
            .padding().background(ColorManager.accentColor).foregroundColor(.white).cornerRadius(10)
        }
        .padding().frame(maxWidth: .infinity).background(ColorManager.cardBackground.opacity(0.7)).cornerRadius(12).padding(.horizontal)
    }
    
    private func resetToInitialUploadState() { /* ... as before, ensures primaryVideoActualRect is reset ... */
        analysisError = nil
        isVideoBeingUploadedOrProcessed = false
        isFetchingComparisonAnalysis = false
        
        videoVMContainer.viewModel?.cleanup()
        videoVMContainer.viewModel = nil
        primaryVideoServerFrames = nil
        primaryVideoActualRect = .zero // Reset video rect crucial for new uploads
        
        comparisonVideoVMContainer.viewModel?.cleanup()
        comparisonVideoVMContainer.viewModel = nil
        comparisonVideoServerFrames = nil
        
        comparisonAnalysisResult = nil
        isUploadingForComparison = false
        shouldShowProcessedPrimaryVideo = false
        navigateToComparisonView = false
    }

    private func uploadButton(isComparisonUpload: Bool) -> some View { /* ... as before ... */
        UploadButton(
            title: isComparisonUpload ?
                LocalizedStringKey("Upload Second Video for Comparison") :
                LocalizedStringKey(String(format: NSLocalizedString("Upload Your %@ Video", comment: ""), technique.name)),
            iconName: "video.badge.plus"
        ) {
            analysisError = nil
            self.isUploadingForComparison = isComparisonUpload
            if !isComparisonUpload {
                comparisonVideoVMContainer.viewModel?.cleanup(); comparisonVideoVMContainer.viewModel = nil
                comparisonVideoServerFrames = nil; comparisonAnalysisResult = nil
                primaryVideoActualRect = .zero // Reset for new primary video
            }
            showUploadOptions = true
        }
        .padding(.top, 20)
    }
    
    // `processedVideoView` now uses `PoseOverlayView` which takes 3D poses from `viewModel`
    // and the `videoActualRect` for correct 2D projection.
    private func processedVideoView(viewModel: VideoPlayerViewModel, videoActualRect: Binding<CGRect>) -> some View {
        VStack(spacing: 12) {
            Text(LocalizedStringKey("Your Analyzed Video"))
                .font(.headline).foregroundColor(ColorManager.textPrimary)
            
            VideoPlayerRepresentable(player: viewModel.player, videoRect: videoActualRect)
                .frame(height: 300).cornerRadius(12)
                // PoseOverlayView now gets 3D poses from viewModel but renders them in 2D using videoActualRect.
                .overlay(PoseOverlayView(viewModel: viewModel, videoActualRect: videoActualRect.wrappedValue))
                .background(Color.black) // Ensures black bars if video aspect ratio differs from frame.
                .onAppear { viewModel.play() }
                .onDisappear { viewModel.pause() }
            
            HStack { // Playback controls
                Button(action: { viewModel.restart() }) { Image(systemName: "arrow.clockwise.circle.fill") }
            }.font(.title).padding().foregroundColor(ColorManager.accentColor)
        }
        .padding(.horizontal)
    }

    private func comparisonOptionsView(primaryVM: VideoPlayerViewModel) -> some View { /* ... as before ... */
        VStack(spacing: 16) {
            Text(LocalizedStringKey("Next Steps"))
                .font(.title3).fontWeight(.semibold).foregroundColor(ColorManager.textPrimary)

            uploadButton(isComparisonUpload: true) // Button to upload a comparison video

            if technique.hasModelVideo { // If a model video is available for this technique
                Button {
                    // Ensure primary video's 3D frame data is available before comparing with model
                    guard primaryVideoServerFrames != nil else {
                        analysisError = "Primary video 3D data not available for model comparison."
                        return
                    }
                    handleCompareWithModelVideo(primaryUserVM: primaryVM)
                } label: {
                    Label(LocalizedStringKey("Compare with Model Video"), systemImage: "person.crop.square.filled.and.at.rectangle")
                        .font(.headline).padding().frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered).tint(ColorManager.accentColor)
            }
            
            Button(LocalizedStringKey("Re-upload Primary Video")) {
                resetToInitialUploadState()
                showUploadOptions = true
            }
            .padding(.top, 10)
            .foregroundColor(ColorManager.accentColor)
        }
        .padding(.horizontal)
    }

    private var keyPointsSection: some View { /* ... as before ... */
        VStack(alignment: .leading, spacing: 16) {
            Text(LocalizedStringKey(String(format: NSLocalizedString("Key Points for %@", comment: ""), technique.name)))
                .font(.headline).foregroundColor(ColorManager.textPrimary)
            KeyPointRow(icon: "figure.play", text: "Start with proper stance, feet shoulder-width apart.")
            KeyPointRow(icon: "hand.raised", text: "Grip the racket with a relaxed, comfortable hold.")
            KeyPointRow(icon: "arrow.up.and.down.and.arrow.left.and.right", text: "Maintain balance throughout the motion.")
            KeyPointRow(icon: "eye", text: "Keep your eye on the shuttle at all times.")
            KeyPointRow(icon: "figure.walk", text: "Follow through with your swing for better control.")
        }
        .padding().background(RoundedRectangle(cornerRadius: 12).fill(ColorManager.cardBackground)).padding(.horizontal)
    }

    // Navigation destination for comparison view.
    // TechniqueComparisonView will receive ViewModels that now manage 3D pose data.
    @ViewBuilder
    private func navigationDestinationView() -> some View {
        if let userVM = videoVMContainer.viewModel,
           let compVM = comparisonVideoVMContainer.viewModel,
           userVM.isVideoReady, compVM.isVideoReady,
           let userFrames = primaryVideoServerFrames, // These are [ServerFrameData] with 3D joints
           let compFrames = comparisonVideoServerFrames { // These are [ServerFrameData] with 3D joints
            
            // Pass the ViewModels (which contain 3D poses) and the 2D analysis result.
            TechniqueComparisonView(
                technique: technique,
                userVideoViewModel: userVM,   // ViewModel now has 3D poses in .poses
                modelVideoViewModel: compVM,  // ViewModel now has 3D poses in .poses
                analysisResult: comparisonAnalysisResult // This is still the 2D swing analysis
            )
        } else {
            ProgressView(LocalizedStringKey("Preparing comparison view..."))
        }
    }

    // MARK: - Video Handling & Analysis Logic
    // These functions now trigger analysis that returns 3D pose data.
    // The VideoPlayerViewModel's `setServerProcessedJoints` method is updated to handle this 3D data.

    private func handlePrimaryVideoSelected(_ url: URL) {
        isVideoBeingUploadedOrProcessed = true
        processingStatusMessage = NSLocalizedString("Uploading & Analyzing Primary Video (3D)...", comment: "") // Updated message
        analysisError = nil
        shouldShowProcessedPrimaryVideo = false
        primaryVideoActualRect = .zero // Reset rect for the new video
        
        videoVMContainer.viewModel?.cleanup() // Clean up previous VM
        videoVMContainer.viewModel = nil
        primaryVideoServerFrames = nil

        // Call analysis service; it now returns VideoAnalysisResponse with 3D jointDataPerFrame.
        analysisService.analyzeVideoByUploading(videoURL: url, dominantSide: "Right")
            .sink(receiveCompletion: { completion in
                self.isVideoBeingUploadedOrProcessed = false
                if case let .failure(error) = completion {
                    self.analysisError = "Failed to analyze primary video (3D): \(error.localizedDescription)"
                    self.shouldShowProcessedPrimaryVideo = false
                }
            }, receiveValue: { response in // response.jointDataPerFrame contains 3D data
                print("Primary video 3D analysis successful. Received \(response.totalFrames) frames.")
                self.primaryVideoServerFrames = response.jointDataPerFrame // Store 3D server frames
                
                let newVM = VideoPlayerViewModel(videoURL: url, videoSource: .primary)
                newVM.setServerProcessedJoints(response.jointDataPerFrame) // VM processes 3D data
                self.videoVMContainer.viewModel = newVM
                
                self.listenForVMReadyAndSetShowFlag(vm: newVM, isPrimary: true)
            })
            .store(in: &cancellables)
    }

    private func handleComparisonVideoSelected(_ url: URL) {
        // Similar to handlePrimaryVideoSelected, but for the comparison video.
        isVideoBeingUploadedOrProcessed = true
        processingStatusMessage = NSLocalizedString("Uploading & Analyzing Comparison Video (3D)...", comment: "")
        analysisError = nil
        
        comparisonVideoVMContainer.viewModel?.cleanup()
        comparisonVideoVMContainer.viewModel = nil
        comparisonVideoServerFrames = nil

        analysisService.analyzeVideoByUploading(videoURL: url, dominantSide: "Right")
            .sink(receiveCompletion: { completion in
                self.isVideoBeingUploadedOrProcessed = false
                if case let .failure(error) = completion {
                    self.analysisError = "Failed to analyze comparison video (3D): \(error.localizedDescription)"
                }
            }, receiveValue: { response in
                print("Comparison video 3D analysis successful. Received \(response.totalFrames) frames.")
                self.comparisonVideoServerFrames = response.jointDataPerFrame
                
                let newCompVM = VideoPlayerViewModel(videoURL: url, videoSource: .secondary)
                newCompVM.setServerProcessedJoints(response.jointDataPerFrame)
                self.comparisonVideoVMContainer.viewModel = newCompVM

                self.listenForVMReadyAndSetShowFlag(vm: newCompVM, isPrimary: false)
            })
            .store(in: &cancellables)
    }
    
    private func handleCompareWithModelVideo(primaryUserVM: VideoPlayerViewModel) {
        // Similar logic, but loads and analyzes the pre-defined model video.
        guard let modelVideoURL = ModelVideoLoader.shared.getModelVideoURL(for: technique.name) else {
            analysisError = "Model video for \(technique.name) not found."
            return
        }
        // Ensure primary user video's 3D data is available.
        guard self.primaryVideoServerFrames != nil else {
            analysisError = "Primary video 3D data is missing for model comparison."
            return
        }
        
        isVideoBeingUploadedOrProcessed = true
        processingStatusMessage = NSLocalizedString("Processing Model Video (3D)...", comment: "")
        analysisError = nil

        comparisonVideoVMContainer.viewModel?.cleanup()
        comparisonVideoVMContainer.viewModel = nil
        comparisonVideoServerFrames = nil
        
        analysisService.analyzeVideoByUploading(videoURL: modelVideoURL, dominantSide: "Right")
            .sink(receiveCompletion: { completion in
                self.isVideoBeingUploadedOrProcessed = false
                if case let .failure(error) = completion {
                    self.analysisError = "Failed to analyze model video (3D): \(error.localizedDescription)"
                }
            }, receiveValue: { response in
                print("Model video 3D analysis successful. Received \(response.totalFrames) frames.")
                self.comparisonVideoServerFrames = response.jointDataPerFrame
                
                let newModelVM = VideoPlayerViewModel(videoURL: modelVideoURL, videoSource: .secondary)
                newModelVM.setServerProcessedJoints(response.jointDataPerFrame)
                self.comparisonVideoVMContainer.viewModel = newModelVM
                
                self.listenForVMReadyAndSetShowFlag(vm: newModelVM, isPrimary: false)
            })
            .store(in: &cancellables)
    }

    // `listenForVMReadyAndSetShowFlag` remains largely the same.
    // It waits for VideoPlayerViewModel to be ready and have poses (which are now 3D).
    private func listenForVMReadyAndSetShowFlag(vm: VideoPlayerViewModel, isPrimary: Bool) {
        var readyCancellable: AnyCancellable?
        readyCancellable = vm.$isVideoReady
            .combineLatest(vm.$accumulatedPoses.map { !$0.isEmpty }) // accumulatedPoses is now 3D
            .filter { $0.0 && $0.1 } // Video ready AND poses available
            .first() // Only take the first time this condition is met
            .sink { [weak vmInstance = vm] _ in // Capture vm weakly
                guard vmInstance != nil else {
                    readyCancellable?.cancel()
                    return
                }
                if isPrimary {
                    print("Primary VM (handling 3D data) is ready and poses are set.")
                    self.shouldShowProcessedPrimaryVideo = true
                } else { // This is for comparison or model video
                    print("Comparison/Model VM (handling 3D data) is ready and poses are set.")
                    // Slight delay before triggering comparison to ensure UI updates settle.
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        self.triggerTechniqueComparison()
                    }
                }
                readyCancellable?.cancel() // Clean up the cancellable
            }
        if let rc = readyCancellable { self.cancellables.insert(rc) }
    }

    // `triggerTechniqueComparison` sends 3D frame data to the server.
    // The server performs 2D swing analysis and returns ComparisonResult.
    private func triggerTechniqueComparison() {
        guard let userFrames = primaryVideoServerFrames, // These are [ServerFrameData] with 3D joints
              let modelFrames = comparisonVideoServerFrames else { // These are [ServerFrameData] with 3D joints
            analysisError = "One or both 3D video data sets are missing for comparison."
            isFetchingComparisonAnalysis = false; return
        }
        // Ensure ViewModels are ready.
        guard let userVM = videoVMContainer.viewModel, userVM.isVideoReady,
              let modelVM = comparisonVideoVMContainer.viewModel, modelVM.isVideoReady else {
            analysisError = "Video players not ready for comparison (using 3D data).";
            isFetchingComparisonAnalysis = false; return
        }

        isFetchingComparisonAnalysis = true; analysisError = nil
        
        // Call service: it sends 3D frames, but server's swing analysis logic is still 2D.
        analysisService.requestTechniqueComparison(
            userFrames: userFrames,
            modelFrames: modelFrames,
            dominantSide: "Right"
        )
        .sink(receiveCompletion: { completion in
            self.isFetchingComparisonAnalysis = false
            if case let .failure(error) = completion {
                self.analysisError = "Technique comparison (from 3D data input) failed: \(error.localizedDescription)"
                 print("Comparison request error: \(error)")
            }
        }, receiveValue: { result in // `result` is ComparisonResult based on 2D server-side analysis
            self.comparisonAnalysisResult = result
            self.navigateToComparisonView = true // Trigger navigation
            print("Technique comparison successful (from 3D data input). Navigating.")
        })
        .store(in: &cancellables)
    }
}

class VideoVMContainer: ObservableObject { @Published var viewModel: VideoPlayerViewModel? }
