// MoveInsight/TechniqueComparisonView.swift
import SwiftUI
import AVKit
import Combine
import SceneKit // Import SceneKit for SceneView3D
import simd     // For SIMD3 used in poses

// Define ComparisonResult struct to match server's JSON response.
// This struct holds the results of the 2D swing analysis performed by the server.
struct ComparisonResult: Codable, Identifiable {
    // Add Identifiable conformance if needed, e.g., for use in ForEach directly with objects.
    // If not directly used in a ForEach that requires Identifiable on the struct itself,
    // `id` can be omitted, but ensure keys in ForEach loops are handled appropriately.
    let id = UUID() // Provides a unique ID, useful if these results are part of a list.
    
    let userScore: Double         // The user's overall technique score.
    let referenceScore: Double    // The model/reference performer's overall score.
    let similarity: [String: Bool] // Dictionary indicating if user matched model on specific criteria.
    let userDetails: [String: Bool]    // Detailed breakdown of user's performance on each criterion.
    let referenceDetails: [String: Bool] // Detailed breakdown of model's performance.

    // CodingKeys to map Swift property names to JSON keys from the server.
    enum CodingKeys: String, CodingKey {
        case userScore = "user_score"           // Maps "user_score" JSON key to userScore.
        case referenceScore = "reference_score" // Maps "reference_score" JSON key to referenceScore.
        case similarity                         // Assumes JSON key is "similarity".
        case userDetails = "user_details"       // Maps "user_details" JSON key to userDetails.
        case referenceDetails = "reference_details" // Maps "reference_details" JSON key to referenceDetails.
    }
}


// Example FeedbackItem if not defined elsewhere (for compilation):
 struct FeedbackItem: View {
     let title: String
     let description: String
     let score: Int // Represents a visual score (e.g., 0-100) for the item color

     var body: some View {
         HStack(alignment: .center, spacing: 16) {
             ZStack { // Score circle
                 Circle().fill(scoreColor.opacity(0.2)).frame(width: 40, height: 40)
                 Text("\(score)%").font(.system(size: 12, weight: .bold)).foregroundColor(scoreColor)
             }
             VStack(alignment: .leading, spacing: 4) { // Feedback text
                 Text(title).font(.subheadline).foregroundColor(ColorManager.textPrimary)
                 Text(description).font(.caption).foregroundColor(ColorManager.textSecondary).lineLimit(nil)
             }
             Spacer()
         }
     }
     private var scoreColor: Color { // Color based on score
         if score >= 90 { return .green }
         else if score >= 75 { return .yellow }
         else if score >= 60 { return .orange }
         else { return .red }
     }
 }


// MARK: - Main Comparison View
struct TechniqueComparisonView: View {
    let technique: BadmintonTechnique
    // ViewModels now manage 3D pose data internally.
    @ObservedObject var userVideoViewModel: VideoPlayerViewModel
    @ObservedObject var modelVideoViewModel: VideoPlayerViewModel
    
    // UI State
    @State private var comparisonMode: ComparisonMode = .sideBySide // Default comparison mode
    @State private var selectedReportTab: ReportTab = .overview     // Default report tab
    
    // Analysis Data (ComparisonResult is based on 2D swing analysis from server)
    @State var analysisResult: ComparisonResult?
    @State private var isAnalyzing = false // For local loading indicators if re-fetching
    @State private var analysisError: String? = nil
    @State private var cancellables = Set<AnyCancellable>()

    // State for video rectangles, used by the 2D side-by-side overlay.
    @State private var userVideoActualRect: CGRect = .zero
    @State private var modelVideoActualRect: CGRect = .zero

    // Delegate for SceneView3D to manage and persist camera state.
    @StateObject private var sceneDelegate = SceneKitViewDelegate()
    
    // Enum for switching between comparison modes.
    enum ComparisonMode {
        case sideBySide  // Shows 2D video players with 2D pose overlays.
        case overlay3D   // Shows 3D skeletons using SceneView3D.
    }
    
    // Enum for switching between report tabs.
    enum ReportTab {
        case overview
        case technical
    }
    
    var body: some View {
        ZStack {
            ColorManager.background.ignoresSafeArea() // Apply background color.
            
            ScrollView {
                VStack(spacing: 20) { // Adjusted spacing
                    // Header text for the technique being analyzed.
                    Text(LocalizedStringKey(String(format: NSLocalizedString("%@ Analysis", comment: ""), technique.name)))
                        .font(.title2).bold().foregroundColor(ColorManager.textPrimary).padding(.top, 16)
                    
                    // Picker to switch between "Side by Side" (2D) and "3D Overlay" modes.
                    Picker("Comparison Mode", selection: $comparisonMode) {
                        Text(LocalizedStringKey("Side by Side")).tag(ComparisonMode.sideBySide)
                        Text(LocalizedStringKey("3D Overlay")).tag(ComparisonMode.overlay3D)
                    }
                    .pickerStyle(SegmentedPickerStyle()).padding(.horizontal) // Use standard padding
                    
                    // Conditional view based on the selected comparison mode.
                    if comparisonMode == .sideBySide {
                        sideBySideComparisonView // Displays 2D video players with overlays.
                    } else { // .overlay3D
                        // Display 3D skeletons using SceneView3D.
                        SceneView3D(
                            userPose3D: userVideoViewModel.poses, // Pass current 3D pose from ViewModel
                            modelPose3D: modelVideoViewModel.poses, // Pass current 3D pose from ViewModel
                            bodyConnections: HumanBodyJoints, // Defined in BodyPoseTypes.swift
                            sceneDelegate: sceneDelegate        // Pass delegate for camera state
                        )
                        .frame(height: 400) // Define a suitable frame for the 3D view.
                        .cornerRadius(12)
                        .overlay(RoundedRectangle(cornerRadius: 12).stroke(ColorManager.accentColor.opacity(0.5), lineWidth: 1)) // Subtle border
                        .padding(.horizontal)
                        .onAppear { // Ensure videos are playing for SceneView3D to receive pose updates.
                            if !userVideoViewModel.isPlaying { userVideoViewModel.play() }
                            if !modelVideoViewModel.isPlaying { modelVideoViewModel.play() }
                        }
                        // Playback controls are shared for both modes now.
                        playbackControls().padding(.top, 8)
                    }
                    
                    // Picker for switching between "Overview" and "Technical" report tabs.
                    Picker("Report Type", selection: $selectedReportTab) {
                        Text(LocalizedStringKey("Overview")).tag(ReportTab.overview)
                        Text(LocalizedStringKey("Technical")).tag(ReportTab.technical)
                    }
                    .pickerStyle(SegmentedPickerStyle()).padding(.horizontal).padding(.top, 10) // Added top padding
                    
                    // Display content based on the selected report tab.
                    switch selectedReportTab {
                    case .overview: overviewAnalysisSection
                    case .technical: technicalAnalysisSection
                    }
                }
                .padding(.bottom, 32)
            }
        }
        .navigationTitle(LocalizedStringKey("Technique Analysis"))
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            // Mute the model/secondary video by default. User's video is unmuted.
            modelVideoViewModel.player.isMuted = true
            userVideoViewModel.player.isMuted = false
            
            // Auto-play videos when the view appears.
            userVideoViewModel.play()
            modelVideoViewModel.play()

            // Log if analysisResult is nil (it should be passed from TechniqueDetailView).
            if analysisResult == nil && !isAnalyzing {
                 print("TechniqueComparisonView: analysisResult is nil on appear. Ensure it's passed from parent.")
            }
        }
        .onDisappear {
            // Pause videos and clean up Combine cancellables when the view disappears.
            userVideoViewModel.pause()
            modelVideoViewModel.pause()
            cancellables.forEach { $0.cancel() }
            cancellables.removeAll()
        }
    }
    
    // View for side-by-side 2D video comparison with overlays.
    private var sideBySideComparisonView: some View {
        VStack(spacing: 12) { // Adjusted spacing
            HStack(spacing: 8) {
                // User's video player card.
                videoPlayerCard(
                    title: LocalizedStringKey("Your Technique"),
                    viewModel: userVideoViewModel,
                    borderColor: .blue,
                    videoActualRectBinding: $userVideoActualRect // Binding for video rect
                )
                // Model's video player card.
                videoPlayerCard(
                    title: LocalizedStringKey("Model Technique"),
                    viewModel: modelVideoViewModel,
                    borderColor: .red,
                    videoActualRectBinding: $modelVideoActualRect // Binding for video rect
                )
            }
            .padding(.horizontal) // Use standard padding
            playbackControls().padding(.top, 8) // Shared playback controls
        }
    }

    // Helper to create a single video player card with its 2D overlay.
    private func videoPlayerCard(
        title: LocalizedStringKey,
        viewModel: VideoPlayerViewModel, // ViewModel now contains 3D poses
        borderColor: Color,
        videoActualRectBinding: Binding<CGRect> // Binding for the video's actual rect
    ) -> some View {
        VStack(spacing: 4) { // Reduced spacing
            Text(title).font(.caption).padding(.bottom, 2).foregroundColor(ColorManager.textPrimary)
            VideoPlayerRepresentable(player: viewModel.player, videoRect: videoActualRectBinding)
                .frame(height: 250) // Adjusted height
                .cornerRadius(10) // Slightly less corner radius
                .overlay(
                    // PoseOverlayView takes 3D poses from viewModel and renders a 2D projection
                    // using the videoActualRect for correct scaling and positioning.
                    PoseOverlayView(viewModel: viewModel, videoActualRect: videoActualRectBinding.wrappedValue)
                )
                .background(Color.black) // Ensure black bars for letter/pillarboxing.
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(borderColor, lineWidth: 1.5)) // Adjusted border
        }
        .frame(maxWidth: .infinity)
    }

    // Shared playback controls for both 2D and 3D modes.
    private func playbackControls() -> some View {
        HStack(spacing: 25) { // Increased spacing for better touch targets
            // Restart button: restarts both videos and plays them.
            Button(action: {
                userVideoViewModel.restart(); modelVideoViewModel.restart()
                userVideoViewModel.play(); modelVideoViewModel.play()
            }) {
                Image(systemName: "backward.end.fill")
                    .imageScale(.large) // Make icon larger
            }
            
            // Play/Pause button: toggles playback for both videos.
            Button(action: {
                if userVideoViewModel.isPlaying {
                    userVideoViewModel.pause(); modelVideoViewModel.pause()
                } else {
                    userVideoViewModel.play(); modelVideoViewModel.play()
                }
            }) {
                Image(systemName: userVideoViewModel.isPlaying ? "pause.fill" : "play.fill")
                    .imageScale(.large) // Make icon larger
            }
        }
        .font(.title2) // Apply font to the HStack for consistent icon sizing if not overridden by imageScale
        .padding(.vertical, 8) // Add some vertical padding
        .foregroundColor(ColorManager.accentColor)
    }
    
    // MARK: - Analysis Display Sections
    // These sections display the 2D swing analysis results from `analysisResult`.
    // Their structure and content remain largely the same as before the 3D update,
    // as `ComparisonResult` is still based on 2D analysis.

    private var overviewAnalysisSection: some View {
        VStack(alignment: .leading, spacing: 16) { // Adjusted spacing
            Text(LocalizedStringKey("Analysis & Feedback")).font(.headline).foregroundColor(ColorManager.textPrimary)
            
            if isAnalyzing { loadingView(message: LocalizedStringKey("Analyzing your technique...")) }
            else if let error = analysisError { errorDisplayView(error) }
            else if let result = analysisResult { // Now `result` is of type ComparisonResult
                techniqueScoreView(score: result.userScore, title: LocalizedStringKey("Overall Technique Score"), subtitle: LocalizedStringKey("Compared to model performance"))
                feedbackDetailsView(details: result.userDetails) // userDetails from ComparisonResult
            } else { noAnalysisDataView() }
        }.padding(.horizontal)
    }

    private var technicalAnalysisSection: some View {
        VStack(alignment: .leading, spacing: 16) { // Adjusted spacing
            Text(LocalizedStringKey("Technical Report")).font(.headline).foregroundColor(ColorManager.textPrimary)

            if isAnalyzing { loadingView(message: LocalizedStringKey("Loading technical report...")) }
            else if let error = analysisError { errorDisplayView(error) }
            else if let result = analysisResult { // Now `result` is of type ComparisonResult
                scoresComparisonView(userScore: result.userScore, referenceScore: result.referenceScore)
                technicalElementsView(similarity: result.similarity, userDetails: result.userDetails, referenceDetails: result.referenceDetails)
                improvementSuggestionsView(userDetails: result.userDetails)
            } else { noAnalysisDataView(message: LocalizedStringKey("Technical analysis data not available.")) }
        }.padding(.horizontal)
    }
    
    // MARK: - Helper Views for Analysis Display (Implementations assumed from previous context or standard UI)
    private func loadingView(message: LocalizedStringKey) -> some View {
        HStack { Spacer(); VStack(spacing: 16) {
            ProgressView().progressViewStyle(CircularProgressViewStyle(tint: ColorManager.accentColor)).scaleEffect(1.5)
            Text(message).font(.subheadline).foregroundColor(ColorManager.textPrimary)
        }.padding(.vertical, 40); Spacer() }
    }
    private func errorDisplayView(_ error: String) -> some View {
        VStack(alignment: .center, spacing: 16) {
            Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 40)).foregroundColor(.orange)
            Text(LocalizedStringKey("Analysis Error")).font(.headline).foregroundColor(ColorManager.textPrimary)
            Text(error).font(.subheadline).foregroundColor(ColorManager.textSecondary).multilineTextAlignment(.center).padding(.horizontal)
            Button(LocalizedStringKey("Retry")) {
                print("Retry tapped. Implement re-fetch logic if this view is responsible.")
            }
            .padding().buttonStyle(.borderedProminent).tint(ColorManager.accentColor)
        }
        .frame(maxWidth: .infinity).padding(20).background(ColorManager.cardBackground.opacity(0.7)).cornerRadius(12).padding(.horizontal)
    }
    private func noAnalysisDataView(message: LocalizedStringKey = LocalizedStringKey("Analysis data not available. Please ensure the video was processed.")) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "info.circle").font(.largeTitle).foregroundColor(ColorManager.textSecondary)
            Text(message).font(.subheadline).foregroundColor(ColorManager.textSecondary)
                .padding(.horizontal, 20).multilineTextAlignment(.center)
            Button(LocalizedStringKey("Refresh Analysis")) {
                print("Refresh Analysis Tapped. Implement re-fetch logic.")
            }
            .padding().buttonStyle(.bordered).tint(ColorManager.accentColor)
        }
        .padding(.vertical, 30).frame(maxWidth: .infinity)
    }
    private func techniqueScoreView(score: Double, title: LocalizedStringKey, subtitle: LocalizedStringKey) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.subheadline).fontWeight(.medium).foregroundColor(ColorManager.textPrimary)
                Text(subtitle).font(.caption).foregroundColor(ColorManager.textSecondary)
            }
            Spacer()
            scoreRingView(score: score, color: ColorManager.accentColor, size: 70)
        }
        .padding().background(ColorManager.cardBackground.opacity(0.8)).cornerRadius(10)
    }
    private func feedbackDetailsView(details: [String: Bool]) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(LocalizedStringKey("Key Technique Elements")).font(.subheadline).fontWeight(.medium).foregroundColor(ColorManager.textPrimary)
            ForEach(Array(details.keys.sorted()), id: \.self) { key in
                let passed = details[key] ?? false
                FeedbackItem(title: formatRuleName(key), description: getDescription(for: key, passed: passed), score: passed ? 95 : 65 )
            }
        }
        .padding().background(ColorManager.cardBackground.opacity(0.8)).cornerRadius(10)
    }
    private func scoresComparisonView(userScore: Double, referenceScore: Double) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(LocalizedStringKey("Scores: You vs. Model")).font(.subheadline).fontWeight(.medium).foregroundColor(ColorManager.textPrimary)
            HStack(spacing: 16) {
                scoreRingView(title: LocalizedStringKey("Your Score"), score: userScore, color: .blue, size: 65)
                scoreRingView(title: LocalizedStringKey("Model Score"), score: referenceScore, color: .red, size: 65)
            }.frame(maxWidth: .infinity)
        }
        .padding().background(ColorManager.cardBackground.opacity(0.8)).cornerRadius(10)
    }
    private func scoreRingView(title: LocalizedStringKey? = nil, score: Double, color: Color, size: CGFloat) -> some View {
        VStack(alignment: .center, spacing: 6) {
            if let title = title { Text(title).font(.caption2).foregroundColor(ColorManager.textSecondary) }
            ZStack {
                Circle().stroke(color.opacity(0.25), lineWidth: size * 0.1).frame(width: size, height: size)
                Circle().trim(from: 0, to: CGFloat(score / 100.0))
                    .stroke(color, style: StrokeStyle(lineWidth: size * 0.1, lineCap: .round))
                    .frame(width: size, height: size).rotationEffect(.degrees(-90))
                Text("\(Int(score))").font(.system(size: size * 0.3, weight: .semibold)).foregroundColor(ColorManager.textPrimary)
            }
        }.frame(maxWidth: .infinity)
    }
    private func technicalElementsView(similarity: [String: Bool], userDetails: [String: Bool], referenceDetails: [String: Bool]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(LocalizedStringKey("Technical Elements Breakdown")).font(.subheadline).fontWeight(.medium).foregroundColor(ColorManager.textPrimary)
            HStack {
                Text(LocalizedStringKey("Element")).font(.caption.weight(.semibold)).foregroundColor(ColorManager.textSecondary).frame(maxWidth: .infinity, alignment: .leading)
                Text(LocalizedStringKey("You")).font(.caption.weight(.semibold)).foregroundColor(ColorManager.textSecondary).frame(width: 50, alignment: .center)
                Text(LocalizedStringKey("Model")).font(.caption.weight(.semibold)).foregroundColor(ColorManager.textSecondary).frame(width: 50, alignment: .center)
            }.padding(.bottom, 2)
            ForEach(Array(similarity.keys.sorted()), id: \.self) { key in
                let userPassed = userDetails[key] ?? false; let modelPassed = referenceDetails[key] ?? false
                HStack {
                    Text(formatRuleName(key)).font(.caption).foregroundColor(ColorManager.textPrimary).frame(maxWidth: .infinity, alignment: .leading)
                    Image(systemName: userPassed ? "checkmark.circle.fill" : "xmark.circle.fill").foregroundColor(userPassed ? .green : .orange).frame(width: 50, alignment: .center)
                    Image(systemName: modelPassed ? "checkmark.circle.fill" : "xmark.circle.fill").foregroundColor(modelPassed ? .green : .orange).frame(width: 50, alignment: .center)
                }.padding(.vertical, 4).background((userPassed == modelPassed) ? Color.clear : Color.yellow.opacity(0.1)).cornerRadius(3)
            }
        }
        .padding().background(ColorManager.cardBackground.opacity(0.8)).cornerRadius(10)
    }
    private func improvementSuggestionsView(userDetails: [String: Bool]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(LocalizedStringKey("Improvement Suggestions")).font(.subheadline).fontWeight(.medium).foregroundColor(ColorManager.textPrimary)
            let improvementAreas = userDetails.filter { !$0.value }.keys.sorted()
            if improvementAreas.isEmpty {
                Text(LocalizedStringKey("Excellent! All key technical elements are performed correctly.")).font(.caption).foregroundColor(.green).padding(.vertical, 4)
            } else {
                ForEach(improvementAreas, id: \.self) { key in
                    HStack(alignment: .top) {
                        Image(systemName: "exclamationmark.circle").foregroundColor(.orange).font(.caption)
                        Text(getDescription(for: key, passed: false)).font(.caption).foregroundColor(ColorManager.textPrimary)
                    }
                }
            }
        }
        .padding().background(ColorManager.cardBackground.opacity(0.8)).cornerRadius(10)
    }

    // Helper functions for formatting text.
    private func formatRuleName(_ rule: String) -> String {
        let localizedKey = NSLocalizedString(rule, comment: "Technical term from server (e.g., shoulder_abduction)")
        return localizedKey == rule ? rule.replacingOccurrences(of: "_", with: " ").capitalized : localizedKey
    }
    private func getDescription(for rule: String, passed: Bool) -> String {
        let baseMessage = formatRuleName(rule)
        let formatKey = passed ? "%@: Well done!" : "%@: Focus on improving this aspect. Check tutorials for guidance."
        return String(format: NSLocalizedString(formatKey, comment: "Feedback string format"), baseMessage)
    }
}
