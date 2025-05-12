import SwiftUI
import AVKit
import Combine

struct TechniqueDetailView: View {
    let technique: BadmintonTechnique
    
    // State to track the user's current progress
    @State private var uploadedVideoURL: URL?
    @State private var videoVM: VideoPlayerViewModel?
    @State private var showUploadOptions = false
    @State private var isVideoBeingProcessed = false
    @State private var isForComparison = false // Track if this is for comparison
    
    // Cancellables for video ready listener
    @State private var cancellables = Set<AnyCancellable>()
    
    // Added state for navigation to comparison view
    @State private var navigateToComparison = false
    @State private var modelVM: VideoPlayerViewModel?
    @State private var secondVideoVM: VideoPlayerViewModel?
    
    var body: some View {
        ZStack {
            ColorManager.background.ignoresSafeArea()
            
            ScrollView {
                VStack(spacing: 24) {
                    // Technique header
                    VStack(spacing: 12) {
                        ZStack {
                            Circle()
                                .fill(ColorManager.accentColor.opacity(0.2))
                                .frame(width: 80, height: 80)
                            
                            Image(systemName: technique.iconName)
                                .font(.system(size: 36))
                                .foregroundColor(ColorManager.accentColor)
                        }
                        
                        Text(technique.name)
                            .font(.title)
                            .foregroundColor(ColorManager.textPrimary)
                            .multilineTextAlignment(.center)
                        
                        Text(technique.description)
                            .font(.body)
                            .foregroundColor(ColorManager.textSecondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 24)
                    }
                    .padding(.top, 24)
                    
                    // Show loading indicator while video is processing
                    if isVideoBeingProcessed {
                        VStack(spacing: 16) {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: ColorManager.accentColor))
                                .scaleEffect(1.5)
                            
                            Text("Processing video...")
                                .foregroundColor(ColorManager.textPrimary)
                        }
                        .frame(height: 300)
                        .frame(maxWidth: .infinity)
                        .background(Color.black.opacity(0.05))
                        .cornerRadius(12)
                        .padding(.horizontal)
                    }
                    // Video preview if uploaded and ready
                    else if let videoVM = videoVM, videoVM.isVideoReady {
                        VStack(spacing: 12) {
                            Text("Your Video")
                                .font(.headline)
                                .foregroundColor(ColorManager.textPrimary)
                            
                            VideoPlayerRepresentable(player: videoVM.player, videoRect: .constant(CGRect()))
                                .frame(height: 300)
                                .cornerRadius(12)
                                .onAppear {
                                    videoVM.play()
                                }
                                .onDisappear {
                                    videoVM.pause()
                                }
                        }
                        .padding(.horizontal)
                        
                        // Options after video is processed and ready
                        VStack(spacing: 20) {
                            Text("What would you like to do next?")
                                .font(.title3)
                                .fontWeight(.bold)
                                .foregroundColor(ColorManager.textPrimary)
                                .padding(.top, 20)
                            
                            HStack(spacing: 16) {
                                Button(action: {
                                    // Navigate to upload a second video for comparison
                                    isForComparison = true
                                    openTechniqueVideoUpload(forComparison: true)
                                }) {
                                    VStack(spacing: 12) {
                                        Image(systemName: "plus.viewfinder")
                                            .font(.system(size: 30))
                                        Text("Upload Second Video")
                                            .font(.headline)
                                            .multilineTextAlignment(.center)
                                    }
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 20)
                                    .background(ColorManager.cardBackground)
                                    .foregroundColor(ColorManager.textPrimary)
                                    .cornerRadius(12)
                                }
                                
                                Button(action: {
                                    // Compare with model video
                                    compareWithModelVideo()
                                }) {
                                    VStack(spacing: 12) {
                                        Image(systemName: "person.fill.viewfinder")
                                            .font(.system(size: 30))
                                        Text("Compare with Model")
                                            .font(.headline)
                                            .multilineTextAlignment(.center)
                                    }
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 20)
                                    .background(ColorManager.accentColor)
                                    .foregroundColor(.white)
                                    .cornerRadius(12)
                                }
                                .disabled(!technique.hasModelVideo)
                                .opacity(technique.hasModelVideo ? 1.0 : 0.5)
                            }
                            .padding(.horizontal)
                        }
                        .padding(.horizontal)
                    } else {
                        // Show upload button if no video is uploaded yet
                        UploadButton(title: "Upload Your \(technique.name) Video", iconName: "video.badge.plus") {
                            isForComparison = false
                            openTechniqueVideoUpload(forComparison: false)
                        }
                        .padding(.top, 20)
                    }
                    
                    // Tips section
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Key Points for \(technique.name)")
                            .font(.headline)
                            .foregroundColor(ColorManager.textPrimary)
                        
                        KeyPointRow(icon: "figure.play", text: "Start with proper stance, feet shoulder-width apart")
                        KeyPointRow(icon: "hand.raised", text: "Grip the racket with a relaxed, comfortable hold")
                        KeyPointRow(icon: "arrow.up.and.down.and.arrow.left.and.right", text: "Maintain balance throughout the motion")
                        KeyPointRow(icon: "eye", text: "Keep your eye on the shuttle at all times")
                        KeyPointRow(icon: "figure.walk", text: "Follow through with your swing for better control")
                    }
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(ColorManager.cardBackground)
                    )
                    .padding(.horizontal)
                }
                .padding(.bottom, 32)
            }
        }
        .navigationTitle(technique.name)
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showUploadOptions) {
            TechniqueVideoUploadView(technique: technique, isComparison: isForComparison) { videoURL in
                if let url = videoURL {
                    if isForComparison {
                        // Handle second video upload - go straight to comparison
                        processSecondVideo(url)
                    } else {
                        // Handle first video upload
                        self.uploadedVideoURL = url
                        processUploadedVideo(url)
                    }
                    
                    // We'll dismiss the sheet after video is fully processed
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                        showUploadOptions = false
                    }
                }
            }
        }
        .onChange(of: showUploadOptions) { isShowing in
            if !isShowing && isVideoBeingProcessed {
                print("Upload sheet dismissed, continuing to monitor video processing state")
            }
        }
        .background(
            // Hidden navigation link for comparison view
            NavigationLink(destination:
                        Group {
                            if isForComparison, let userVM = videoVM, let secondVM = secondVideoVM {
                                // Two user videos
                                TechniqueComparisonView(
                                    technique: technique,
                                    userVideoViewModel: userVM,
                                    modelVideoViewModel: secondVM
                                )
                            } else if let userVM = videoVM, let modelVM = modelVM {
                                // User video vs model
                                TechniqueComparisonView(
                                    technique: technique,
                                    userVideoViewModel: userVM,
                                    modelVideoViewModel: modelVM
                                )
                            } else {
                                Text("Preparing comparison...")
                            }
                        },
                       isActive: $navigateToComparison) {
                EmptyView()
            }
        )
    }
    
    // Function to open the video upload view
    private func openTechniqueVideoUpload(forComparison: Bool) {
        self.showUploadOptions = true
    }
    
    // Process the uploaded video (first video)
    private func processUploadedVideo(_ url: URL) {
        // Indicate that processing has started
        isVideoBeingProcessed = true
        
        // Create the video view model and begin processing
        videoVM = VideoPlayerViewModel(
            videoURL: url,
            videoSource: .primary
        )
        
        // Set up a publisher to listen for isVideoReady changes
        guard let videoVM = videoVM else { return }
        
        videoVM.$isVideoReady
            .dropFirst() // Skip initial false value
            .sink { isReady in
                if isReady {
                    print("Video is now ready - updating UI")
                    self.isVideoBeingProcessed = false
                }
            }
            .store(in: &cancellables)
        
        print("Started video processing - waiting for it to be ready")
    }
    
    // Process the second uploaded video for comparison
    private func processSecondVideo(_ url: URL) {
        // Create model for second video
        let secondVM = VideoPlayerViewModel(
            videoURL: url,
            videoSource: .secondary
        )
        
        self.secondVideoVM = secondVM
        
        // Wait for the second video to be ready
        secondVM.$isVideoReady
            .dropFirst()
            .sink { isReady in
                if isReady && self.videoVM?.isVideoReady == true {
                    // Navigate to comparison view when both videos are ready
                    DispatchQueue.main.async {
                        self.navigateToComparison = true
                    }
                }
            }
            .store(in: &cancellables)
    }
    
    // Function to compare with model video
    private func compareWithModelVideo() {
        // Use ModelVideoLoader to get the model video
        let modelLoader = ModelVideoLoader.shared
        
        // Get model video for this specific technique
        guard let newModelVM = modelLoader.createModelVideoViewModel(for: technique.name) else {
            // Show an alert to the user
            let alert = UIAlertController(
                title: "Model Video Not Available",
                message: "The model video for \(technique.name) could not be loaded. Please try again later.",
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            
            DispatchQueue.main.async {
                if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                   let rootViewController = windowScene.windows.first?.rootViewController {
                    rootViewController.present(alert, animated: true)
                }
            }
            return
        }
        
        // Store the model view model and navigate
        self.isForComparison = false // Not using second user video
        self.modelVM = newModelVM
        self.navigateToComparison = true
    }
}

// Helper View for Key Points
struct KeyPointRow: View {
    let icon: String
    let text: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 18))
                .foregroundColor(ColorManager.accentColor)
                .frame(width: 24, height: 24)
            
            Text(text)
                .font(.subheadline)
                .foregroundColor(ColorManager.textPrimary)
            
            Spacer()
        }
    }
}
