//import SwiftUI
//import AVKit
//
//struct TechniqueComparisonView: View {
//    let technique: BadmintonTechnique
//    let userVideoViewModel: VideoPlayerViewModel
//    let modelVideoViewModel: VideoPlayerViewModel
//    
//    @State private var comparisonMode: ComparisonMode = .sideBySide
//    @State private var selectedReportTab: ReportTab = .overview
//    
//    enum ComparisonMode {
//        case sideBySide
//        case overlay3D
//    }
//    
//    enum ReportTab {
//        case overview
//        case technical
//        case smash
//        case positioning
//    }
//    
//    var body: some View {
//        ZStack {
//            ColorManager.background.ignoresSafeArea()
//            
//            ScrollView {
//                VStack(spacing: 24) {
//                    // Header
//                    Text("\(technique.name) Comparison")
//                        .font(.title2)
//                        .foregroundColor(ColorManager.textPrimary)
//                        .padding(.top, 16)
//                    
//                    // Comparison Mode Selector
//                    Picker("Comparison Mode", selection: $comparisonMode) {
//                        Text("Side by Side").tag(ComparisonMode.sideBySide)
//                        Text("3D Overlay").tag(ComparisonMode.overlay3D)
//                    }
//                    .pickerStyle(SegmentedPickerStyle())
//                    .padding(.horizontal, 20)
//                    
//                    // Comparison content based on selected mode
//                    if comparisonMode == .sideBySide {
//                        sideBySideComparisonView
//                    } else {
//                        overlay3DComparisonView
//                    }
//                    
//                    // Report Tabs
//                    Picker("Report Type", selection: $selectedReportTab) {
//                        Text("Overview").tag(ReportTab.overview)
//                        Text("Technical").tag(ReportTab.technical)
//                        Text("Smash").tag(ReportTab.smash)
//                        Text("Positioning").tag(ReportTab.positioning)
//                    }
//                    .pickerStyle(SegmentedPickerStyle())
//                    .padding(.horizontal, 20)
//                    .padding(.top, 8)
//                    
//                    // Analysis and Feedback based on selected tab
//                    switch selectedReportTab {
//                    case .overview:
//                        overviewAnalysisSection
//                    case .technical:
//                        technicalAnalysisSection
//                    case .smash:
//                        smashAnalysisSection
//                    case .positioning:
//                        positioningAnalysisSection
//                    }
//                }
//                .padding(.bottom, 32)
//            }
//        }
//        .navigationTitle("Technique Analysis")
//        .navigationBarTitleDisplayMode(.inline)
//        .onAppear {
//            // Ensure videos play when the view appears
//            userVideoViewModel.play()
//            modelVideoViewModel.play()
//        }
//        .onDisappear {
//            // Stop videos when view disappears
//            userVideoViewModel.pause()
//            modelVideoViewModel.pause()
//        }
//    }
//    
//    // Side-by-side comparison view
//    private var sideBySideComparisonView: some View {
//        VStack(spacing: 16) {
//            // Videos
//            HStack(spacing: 8) {
//                // Your video
//                VStack {
//                    Text("Your Technique")
//                        .font(.subheadline)
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    VideoPlayerRepresentable(player: userVideoViewModel.player, videoRect: .constant(CGRect()))
//                        .frame(height: 240)
//                        .cornerRadius(12)
//                        .overlay(
//                            RoundedRectangle(cornerRadius: 12)
//                                .stroke(Color.blue, lineWidth: 2)
//                        )
//                }
//                .frame(maxWidth: .infinity)
//                
//                // Model video
//                VStack {
//                    Text("Model Technique")
//                        .font(.subheadline)
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    VideoPlayerRepresentable(player: modelVideoViewModel.player, videoRect: .constant(CGRect()))
//                        .frame(height: 240)
//                        .cornerRadius(12)
//                        .overlay(
//                            RoundedRectangle(cornerRadius: 12)
//                                .stroke(Color.red, lineWidth: 2)
//                        )
//                }
//                .frame(maxWidth: .infinity)
//            }
//            .padding(.horizontal, 8)
//            
//            // Playback controls
//            HStack {
//                Button(action: {
//                    if userVideoViewModel.isPlaying {
//                        userVideoViewModel.pause()
//                        modelVideoViewModel.pause()
//                    } else {
//                        userVideoViewModel.play()
//                        modelVideoViewModel.play()
//                    }
//                }) {
//                    Image(systemName: userVideoViewModel.isPlaying ? "pause.fill" : "play.fill")
//                        .font(.system(size: 24))
//                        .foregroundColor(ColorManager.accentColor)
//                        .frame(width: 44, height: 44)
//                }
//                
//                Button(action: {
//                    userVideoViewModel.restart()
//                    modelVideoViewModel.restart()
//                }) {
//                    Image(systemName: "arrow.clockwise")
//                        .font(.system(size: 20))
//                        .foregroundColor(ColorManager.textPrimary)
//                        .frame(width: 44, height: 44)
//                }
//            }
//            .padding(.bottom, 8)
//        }
//    }
//    
//    // 3D overlay comparison view
//    private var overlay3DComparisonView: some View {
//        VStack(spacing: 16) {
//            // 3D View
//            CombinedVideo3DView(
//                baseViewModel: userVideoViewModel,
//                overlayViewModel: modelVideoViewModel
//            )
//            .frame(height: 400)
//            .cornerRadius(12)
//            .overlay(
//                RoundedRectangle(cornerRadius: 12)
//                    .stroke(ColorManager.accentColor, lineWidth: 2)
//            )
//            .padding(.horizontal, 16)
//            
//            Text("Blue skeleton: Your technique | Red skeleton: Model technique")
//                .font(.caption)
//                .foregroundColor(ColorManager.textSecondary)
//        }
//    }
//    
//    // Original Analysis and feedback section now called "Overview"
//    private var overviewAnalysisSection: some View {
//        VStack(alignment: .leading, spacing: 20) {
//            Text("Analysis & Feedback")
//                .font(.headline)
//                .foregroundColor(ColorManager.textPrimary)
//                .padding(.horizontal, 20)
//            
//            // Technique score
//            HStack {
//                VStack(alignment: .leading, spacing: 4) {
//                    Text("Overall Technique Score")
//                        .font(.subheadline)
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    Text("Based on comparing your form to the model technique")
//                        .font(.caption)
//                        .foregroundColor(ColorManager.textSecondary)
//                }
//                
//                Spacer()
//                
//                ZStack {
//                    Circle()
//                        .stroke(ColorManager.accentColor.opacity(0.3), lineWidth: 8)
//                        .frame(width: 70, height: 70)
//                    
//                    Circle()
//                        .trim(from: 0, to: 0.78) // 78% score
//                        .stroke(ColorManager.accentColor, style: StrokeStyle(lineWidth: 8, lineCap: .round))
//                        .frame(width: 70, height: 70)
//                        .rotationEffect(.degrees(-90))
//                    
//                    Text("78%")
//                        .font(.system(size: 18, weight: .bold))
//                        .foregroundColor(ColorManager.textPrimary)
//                }
//            }
//            .padding(.horizontal, 20)
//            .padding(.vertical, 16)
//            .background(ColorManager.cardBackground.opacity(0.5))
//            .cornerRadius(12)
//            .padding(.horizontal, 20)
//            
//            // Feedback points
//            VStack(alignment: .leading, spacing: 12) {
//                Text("Improvement Areas")
//                    .font(.subheadline)
//                    .foregroundColor(ColorManager.textPrimary)
//                
//                FeedbackItem(
//                    title: "Arm Extension",
//                    description: "Your arm extension is 15% shorter than recommended",
//                    score: 70
//                )
//                
//                FeedbackItem(
//                    title: "Follow Through",
//                    description: "Your follow through motion completes correctly",
//                    score: 95
//                )
//                
//                FeedbackItem(
//                    title: "Racket Angle",
//                    description: "Your racket angle is 10° more vertical than ideal",
//                    score: 75
//                )
//                
//                FeedbackItem(
//                    title: "Timing",
//                    description: "Your timing is well synchronized with the shuttle",
//                    score: 90
//                )
//            }
//            .padding(20)
//            .background(ColorManager.cardBackground.opacity(0.5))
//            .cornerRadius(12)
//            .padding(.horizontal, 20)
//        }
//    }
//    
//    // NEW: Technical Analysis Section
//    private var technicalAnalysisSection: some View {
//        VStack(alignment: .leading, spacing: 20) {
//            Text("Technical Report")
//                .font(.headline)
//                .foregroundColor(ColorManager.textPrimary)
//                .padding(.horizontal, 20)
//            
//            // Technique usage summary
//            VStack(alignment: .leading, spacing: 16) {
//                Text("Technique Usage Summary")
//                    .font(.subheadline)
//                    .foregroundColor(ColorManager.textPrimary)
//                
//                // Column headers
//                HStack {
//                    Text("Technique Type")
//                        .font(.system(size: 14))
//                        .foregroundColor(ColorManager.textSecondary)
//                        .frame(width: 180, alignment: .leading)
//                    
//                    Spacer()
//                    
//                    Text("Uses")
//                        .font(.system(size: 14))
//                        .foregroundColor(ColorManager.textSecondary)
//                        .frame(width: 40, alignment: .center)
//                    
//                    Text("Quality")
//                        .font(.system(size: 14))
//                        .foregroundColor(ColorManager.textSecondary)
//                        .frame(width: 80, alignment: .trailing)
//                }
//                .padding(.bottom, 8)
//                
//                VStack(alignment: .leading, spacing: 12) {
//                    // Smash stats
//                    HStack {
//                        Text("Smash")
//                            .font(.system(size: 16, weight: .medium))
//                            .foregroundColor(ColorManager.textPrimary)
//                            .frame(width: 180, alignment: .leading)
//                        
//                        Spacer()
//                        
//                        Text("10")
//                            .font(.system(size: 16))
//                            .foregroundColor(ColorManager.textSecondary)
//                            .frame(width: 40, alignment: .center)
//                        
//                        Text("40/100")
//                            .font(.system(size: 16))
//                            .foregroundColor(.orange)
//                            .frame(width: 80, alignment: .trailing)
//                    }
//                    
//                    // Forehand High Clear stats
//                    HStack {
//                        Text("Forehand High Clear")
//                            .font(.system(size: 16, weight: .medium))
//                            .foregroundColor(ColorManager.textPrimary)
//                            .frame(width: 180, alignment: .leading)
//                        
//                        Spacer()
//                        
//                        Text("21")
//                            .font(.system(size: 16))
//                            .foregroundColor(ColorManager.textSecondary)
//                            .frame(width: 40, alignment: .center)
//                        
//                        Text("60/100")
//                            .font(.system(size: 16))
//                            .foregroundColor(.yellow)
//                            .frame(width: 80, alignment: .trailing)
//                    }
//                    
//                    // Backhand High Clear stats
//                    HStack {
//                        Text("Backhand High Clear")
//                            .font(.system(size: 16, weight: .medium))
//                            .foregroundColor(ColorManager.textPrimary)
//                            .frame(width: 180, alignment: .leading)
//                        
//                        Spacer()
//                        
//                        Text("12")
//                            .font(.system(size: 16))
//                            .foregroundColor(ColorManager.textSecondary)
//                            .frame(width: 40, alignment: .center)
//                        
//                        Text("92/100")
//                            .font(.system(size: 16))
//                            .foregroundColor(.green)
//                            .frame(width: 80, alignment: .trailing)
//                    }
//                    
//                    // Drop Shot stats
//                    HStack {
//                        Text("Drop Shot")
//                            .font(.system(size: 16, weight: .medium))
//                            .foregroundColor(ColorManager.textPrimary)
//                            .frame(width: 180, alignment: .leading)
//                        
//                        Spacer()
//                        
//                        Text("12")
//                            .font(.system(size: 16))
//                            .foregroundColor(ColorManager.textSecondary)
//                            .frame(width: 40, alignment: .center)
//                        
//                        Text("92/100")
//                            .font(.system(size: 16))
//                            .foregroundColor(.green)
//                            .frame(width: 80, alignment: .trailing)
//                    }
//                    
//                    // Net Shot stats
//                    HStack {
//                        Text("Net Shot")
//                            .font(.system(size: 16, weight: .medium))
//                            .foregroundColor(ColorManager.textPrimary)
//                            .frame(width: 180, alignment: .leading)
//                        
//                        Spacer()
//                        
//                        Text("5")
//                            .font(.system(size: 16))
//                            .foregroundColor(ColorManager.textSecondary)
//                            .frame(width: 40, alignment: .center)
//                        
//                        Text("92/100")
//                            .font(.system(size: 16))
//                            .foregroundColor(.green)
//                            .frame(width: 80, alignment: .trailing)
//                    }
//                }
//            }
//            .padding(20)
//            .background(ColorManager.cardBackground.opacity(0.5))
//            .cornerRadius(12)
//            .padding(.horizontal, 20)
//            
//            // Summary and recommendations
//            VStack(alignment: .leading, spacing: 12) {
//                Text("Summary & Recommendations")
//                    .font(.subheadline)
//                    .foregroundColor(ColorManager.textPrimary)
//                
//                VStack(alignment: .leading, spacing: 8) {
//                    Text("• Only used smash 10 times with 40% success rate. Recommended to increase training frequency.")
//                        .font(.system(size: 15))
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    Text("• Forehand high clear success rate is 60% (21 attempts).")
//                        .font(.system(size: 15))
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    Text("• Backhand high clear, drop shot, net shot, and push technique scores are above average.")
//                        .font(.system(size: 15))
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    Text("• Focus on improving smash power and accuracy through dedicated practice sessions.")
//                        .font(.system(size: 15))
//                        .foregroundColor(ColorManager.textPrimary)
//                }
//            }
//            .padding(20)
//            .background(ColorManager.cardBackground.opacity(0.5))
//            .cornerRadius(12)
//            .padding(.horizontal, 20)
//        }
//    }
//    
//    // NEW: Smash Analysis Section
//    private var smashAnalysisSection: some View {
//        VStack(alignment: .leading, spacing: 20) {
//            Text("Smash Technique Analysis")
//                .font(.headline)
//                .foregroundColor(ColorManager.textPrimary)
//                .padding(.horizontal, 20)
//            
//            // Movement sequence chart
//            VStack(alignment: .leading, spacing: 12) {
//                Text("Velocity Profile")
//                    .font(.subheadline)
//                    .foregroundColor(ColorManager.textPrimary)
//                
//                Image("velocity_profile_chart") // Would need to be added to assets
//                    .resizable()
//                    .scaledToFit()
//                    .cornerRadius(8)
//                
//                Divider()
//                
//                // Movement sequence breakdown
//                VStack(alignment: .leading, spacing: 8) {
//                    Text("Optimal Movement Sequence:")
//                        .font(.system(size: 16, weight: .medium))
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    HStack(spacing: 0) {
//                        Text("Upper Body")
//                            .font(.system(size: 12))
//                            .foregroundColor(.green)
//                            .padding(.vertical, 4)
//                            .padding(.horizontal, 8)
//                            .background(Color.green.opacity(0.1))
//                            .cornerRadius(4)
//                            .overlay(
//                                RoundedRectangle(cornerRadius: 4)
//                                    .stroke(Color.green, style: StrokeStyle(lineWidth: 1, dash: [3]))
//                            )
//                        
//                        Image(systemName: "arrow.right")
//                            .font(.system(size: 12))
//                            .foregroundColor(ColorManager.textSecondary)
//                            .padding(.horizontal, 4)
//                        
//                        Text("Upper Arm")
//                            .font(.system(size: 12))
//                            .foregroundColor(.green)
//                            .padding(.vertical, 4)
//                            .padding(.horizontal, 8)
//                            .background(Color.green.opacity(0.1))
//                            .cornerRadius(4)
//                            .overlay(
//                                RoundedRectangle(cornerRadius: 4)
//                                    .stroke(Color.green, style: StrokeStyle(lineWidth: 1, dash: [3]))
//                            )
//                        
//                        Image(systemName: "arrow.right")
//                            .font(.system(size: 12))
//                            .foregroundColor(ColorManager.textSecondary)
//                            .padding(.horizontal, 4)
//                        
//                        Text("Forearm")
//                            .font(.system(size: 12))
//                            .foregroundColor(ColorManager.accentColor)
//                            .padding(.vertical, 4)
//                            .padding(.horizontal, 8)
//                            .background(ColorManager.accentColor.opacity(0.1))
//                            .cornerRadius(4)
//                        
//                        Image(systemName: "arrow.right")
//                            .font(.system(size: 12))
//                            .foregroundColor(ColorManager.textSecondary)
//                            .padding(.horizontal, 4)
//                        
//                        Text("Wrist")
//                            .font(.system(size: 12))
//                            .foregroundColor(ColorManager.accentColor)
//                            .padding(.vertical, 4)
//                            .padding(.horizontal, 8)
//                            .background(ColorManager.accentColor.opacity(0.1))
//                            .cornerRadius(4)
//                    }
//                    .padding(.vertical, 8)
//                    
//                    Text("Your Movement Sequence:")
//                        .font(.system(size: 16, weight: .medium))
//                        .foregroundColor(ColorManager.textPrimary)
//                        .padding(.top, 4)
//                    
//                    HStack(spacing: 0) {
//                        Text("Upper Arm")
//                            .font(.system(size: 12))
//                            .foregroundColor(.red)
//                            .padding(.vertical, 4)
//                            .padding(.horizontal, 8)
//                            .background(Color.red.opacity(0.1))
//                            .cornerRadius(4)
//                            .overlay(
//                                RoundedRectangle(cornerRadius: 4)
//                                    .stroke(Color.red, style: StrokeStyle(lineWidth: 1, dash: [3]))
//                            )
//                        
//                        Image(systemName: "arrow.right")
//                            .font(.system(size: 12))
//                            .foregroundColor(ColorManager.textSecondary)
//                            .padding(.horizontal, 4)
//                        
//                        Text("Upper Body")
//                            .font(.system(size: 12))
//                            .foregroundColor(.red)
//                            .padding(.vertical, 4)
//                            .padding(.horizontal, 8)
//                            .background(Color.red.opacity(0.1))
//                            .cornerRadius(4)
//                            .overlay(
//                                RoundedRectangle(cornerRadius: 4)
//                                    .stroke(Color.red, style: StrokeStyle(lineWidth: 1, dash: [3]))
//                            )
//                        
//                        Image(systemName: "arrow.right")
//                            .font(.system(size: 12))
//                            .foregroundColor(ColorManager.textSecondary)
//                            .padding(.horizontal, 4)
//                        
//                        // Missing forearm component - marked in red
//                        Text("Forearm")
//                            .font(.system(size: 12))
//                            .foregroundColor(ColorManager.accentColor)
//                            .padding(.vertical, 4)
//                            .padding(.horizontal, 8)
//                            .background(ColorManager.accentColor.opacity(0.1))
//                            .cornerRadius(4)
//                        
//                        Image(systemName: "arrow.right")
//                            .font(.system(size: 12))
//                            .foregroundColor(ColorManager.textSecondary)
//                            .padding(.horizontal, 4)
//                        
//                        Text("Wrist")
//                            .font(.system(size: 12))
//                            .foregroundColor(ColorManager.accentColor)
//                            .padding(.vertical, 4)
//                            .padding(.horizontal, 8)
//                            .background(ColorManager.accentColor.opacity(0.1))
//                            .cornerRadius(4)
//                    }
//                    .padding(.vertical, 8)
//                }
//            }
//            .padding(20)
//            .background(ColorManager.cardBackground.opacity(0.5))
//            .cornerRadius(12)
//            .padding(.horizontal, 20)
//            
//            // Movement analysis
//            VStack(alignment: .leading, spacing: 12) {
//                Text("Power Generation Analysis")
//                    .font(.subheadline)
//                    .foregroundColor(ColorManager.textPrimary)
//                
//                VStack(alignment: .leading, spacing: 10) {
//                    Text("Your power generation is sequential but missing key component:")
//                        .font(.system(size: 15))
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    Text("• Missing forearm rotation phase reduces shuttle velocity by 12-15%")
//                        .font(.system(size: 15))
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    Text("• Proper forearm pronation helps transfer energy from elbow to wrist")
//                        .font(.system(size: 15))
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    Text("• When elbow rotation is at 150°/s, shuttle peak velocity increases by 12-15%")
//                        .font(.system(size: 15))
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    Text("• Focus on adding deliberate forearm rotation before wrist snap for maximum power")
//                        .font(.system(size: 15))
//                        .foregroundColor(ColorManager.textPrimary)
//                }
//            }
//            .padding(20)
//            .background(ColorManager.cardBackground.opacity(0.5))
//            .cornerRadius(12)
//            .padding(.horizontal, 20)
//        }
//    }
//    
//    // NEW: Positioning Analysis Section
//    private var positioningAnalysisSection: some View {
//        VStack(alignment: .leading, spacing: 20) {
//            Text("Player-Shuttle Positioning")
//                .font(.headline)
//                .foregroundColor(ColorManager.textPrimary)
//                .padding(.horizontal, 20)
//            
//            // Positioning measurements
//            VStack(alignment: .leading, spacing: 12) {
//                Text("Measurements")
//                    .font(.subheadline)
//                    .foregroundColor(ColorManager.textPrimary)
//                
//                HStack(spacing: 20) {
//                    VStack(alignment: .leading, spacing: 8) {
//                        Text("Your Height:")
//                            .font(.system(size: 15))
//                            .foregroundColor(ColorManager.textSecondary)
//                        
//                        HStack {
//                            Text("2.0m")
//                                .font(.system(size: 18, weight: .bold))
//                                .foregroundColor(.red)
//                            
//                            Image(systemName: "arrow.down")
//                                .font(.system(size: 12))
//                                .foregroundColor(.red)
//                        }
//                        
//                        Text("Recommended:")
//                            .font(.system(size: 15))
//                            .foregroundColor(ColorManager.textSecondary)
//                            .padding(.top, 4)
//                        
//                        Text("2.1m")
//                            .font(.system(size: 18, weight: .bold))
//                            .foregroundColor(.green)
//                    }
//                    
//                    Divider()
//                        .frame(width: 1, height: 100)
//                        .background(ColorManager.textSecondary.opacity(0.3))
//                    
//                    VStack(alignment: .leading, spacing: 8) {
//                        Text("Your Horizontal Distance:")
//                            .font(.system(size: 15))
//                            .foregroundColor(ColorManager.textSecondary)
//                        
//                        HStack {
//                            Text("60cm")
//                                .font(.system(size: 18, weight: .bold))
//                                .foregroundColor(.red)
//                            
//                            Image(systemName: "arrow.right")
//                                .font(.system(size: 12))
//                                .foregroundColor(.red)
//                        }
//                        
//                        Text("Recommended:")
//                            .font(.system(size: 15))
//                            .foregroundColor(ColorManager.textSecondary)
//                            .padding(.top, 4)
//                        
//                        Text("40-50cm")
//                            .font(.system(size: 18, weight: .bold))
//                            .foregroundColor(.green)
//                    }
//                }
//                .padding(.vertical, 8)
//                
//                // Positioning diagram
//                Image("player_shuttle_distance")
//                    .resizable()
//                    .scaledToFit()
//                    .cornerRadius(8)
//            }
//            .padding(20)
//            .background(ColorManager.cardBackground.opacity(0.5))
//            .cornerRadius(12)
//            .padding(.horizontal, 20)
//            
//            // Positioning analysis and recommendations
//            VStack(alignment: .leading, spacing: 12) {
//                Text("Analysis & Improvement Suggestions")
//                    .font(.subheadline)
//                    .foregroundColor(ColorManager.textPrimary)
//                
//                VStack(alignment: .leading, spacing: 12) {
//                    Text("Impact of current positioning:")
//                        .font(.system(size: 15, weight: .medium))
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    Text("• Hitting point too low and forward prevents full arm extension and power generation")
//                        .font(.system(size: 15))
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    Text("• Shuttle contact angle becomes too flat, reducing control and shot quality")
//                        .font(.system(size: 15))
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    Text("• Wrist and shoulder strain increases due to improper power generation mechanics")
//                        .font(.system(size: 15))
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    Text("Recommended adjustments:")
//                        .font(.system(size: 15, weight: .medium))
//                        .foregroundColor(ColorManager.textPrimary)
//                        .padding(.top, 8)
//                    
//                    Text("• Adjust hitting timing to keep contact point above your body")
//                        .font(.system(size: 15))
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    Text("• Allow natural wrist extension position for optimal power transfer")
//                        .font(.system(size: 15))
//                        .foregroundColor(ColorManager.textPrimary)
//                    
//                    Text("• Decrease horizontal distance to 40-50cm for better leverage and control")
//                        .font(.system(size: 15))
//                        .foregroundColor(ColorManager.textPrimary)
//                }
//            }
//            .padding(20)
//            .background(ColorManager.cardBackground.opacity(0.5))
//            .cornerRadius(12)
//            .padding(.horizontal, 20)
//        }
//    }
//}
//
//// Feedback item component
//struct FeedbackItem: View {
//    let title: String
//    let description: String
//    let score: Int
//    
//    var body: some View {
//        HStack(alignment: .center, spacing: 16) {
//            // Score circle
//            ZStack {
//                Circle()
//                    .fill(scoreColor.opacity(0.2))
//                    .frame(width: 40, height: 40)
//                
//                Text("\(score)%")
//                    .font(.system(size: 12, weight: .bold))
//                    .foregroundColor(scoreColor)
//            }
//            
//            // Feedback text
//            VStack(alignment: .leading, spacing: 4) {
//                Text(title)
//                    .font(.subheadline)
//                    .foregroundColor(ColorManager.textPrimary)
//                
//                Text(description)
//                    .font(.caption)
//                    .foregroundColor(ColorManager.textSecondary)
//            }
//        }
//    }
//    
//    // Color based on the score
//    private var scoreColor: Color {
//        if score >= 90 {
//            return .green
//        } else if score >= 75 {
//            return .yellow
//        } else if score >= 60 {
//            return .orange
//        } else {
//            return .red
//        }
//    }
//}

import SwiftUI
import AVKit
import Combine

struct TechniqueComparisonView: View {
    let technique: BadmintonTechnique
    let userVideoViewModel: VideoPlayerViewModel
    let modelVideoViewModel: VideoPlayerViewModel
    
    @State private var comparisonMode: ComparisonMode = .sideBySide
    @State private var selectedReportTab: ReportTab = .overview
    
    // Analysis state
    @State private var analysisResult: ComparisonResult?
    @State private var isAnalyzing = false
    @State private var analysisError: String? = nil
    @State private var cancellables = Set<AnyCancellable>()
    
    enum ComparisonMode {
        case sideBySide
        case overlay3D
    }
    
    enum ReportTab {
        case overview
        case technical
    }
    
    var body: some View {
        ZStack {
            ColorManager.background.ignoresSafeArea()
            
            ScrollView {
                VStack(spacing: 24) {
                    // Header
                    Text("\(technique.name) Analysis")
                        .font(.title2)
                        .foregroundColor(ColorManager.textPrimary)
                        .padding(.top, 16)
                    
                    // Comparison Mode Selector
                    Picker("Comparison Mode", selection: $comparisonMode) {
                        Text("Side by Side").tag(ComparisonMode.sideBySide)
                        Text("3D Overlay").tag(ComparisonMode.overlay3D)
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .padding(.horizontal, 20)
                    
                    // Comparison content based on selected mode
                    if comparisonMode == .sideBySide {
                        sideBySideComparisonView
                    } else {
                        overlay3DComparisonView
                    }
                    
                    // Report Tabs
                    Picker("Report Type", selection: $selectedReportTab) {
                        Text("Overview").tag(ReportTab.overview)
                        Text("Technical").tag(ReportTab.technical)
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .padding(.horizontal, 20)
                    .padding(.top, 8)
                    
                    // Analysis and Feedback based on selected tab
                    switch selectedReportTab {
                    case .overview:
                        overviewAnalysisSection
                    case .technical:
                        technicalAnalysisSection
                    }
                }
                .padding(.bottom, 32)
            }
        }
        .navigationTitle("Technique Analysis")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            // Ensure videos play when the view appears
            userVideoViewModel.play()
            modelVideoViewModel.play()
            
            // Perform analysis after a short delay to ensure videos are loaded
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                performAnalysis()
            }
        }
        .onDisappear {
            // Stop videos when view disappears
            userVideoViewModel.pause()
            modelVideoViewModel.pause()
            
            // Cancel any pending analysis
            cancellables.forEach { $0.cancel() }
            cancellables.removeAll()
        }
    }
    
    // Side-by-side comparison view
    private var sideBySideComparisonView: some View {
        VStack(spacing: 16) {
            // Videos
            HStack(spacing: 8) {
                // Your video
                VStack {
                    Text("Your Technique")
                        .font(.subheadline)
                        .foregroundColor(ColorManager.textPrimary)
                    
                    VideoPlayerRepresentable(player: userVideoViewModel.player, videoRect: .constant(CGRect()))
                        .frame(height: 240)
                        .cornerRadius(12)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.blue, lineWidth: 2)
                        )
                }
                .frame(maxWidth: .infinity)
                
                // Model video
                VStack {
                    Text("Model Technique")
                        .font(.subheadline)
                        .foregroundColor(ColorManager.textPrimary)
                    
                    VideoPlayerRepresentable(player: modelVideoViewModel.player, videoRect: .constant(CGRect()))
                        .frame(height: 240)
                        .cornerRadius(12)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.red, lineWidth: 2)
                        )
                }
                .frame(maxWidth: .infinity)
            }
            .padding(.horizontal, 8)
            
            // Playback controls
            HStack {
                Button(action: {
                    if userVideoViewModel.isPlaying {
                        userVideoViewModel.pause()
                        modelVideoViewModel.pause()
                    } else {
                        userVideoViewModel.play()
                        modelVideoViewModel.play()
                    }
                }) {
                    Image(systemName: userVideoViewModel.isPlaying ? "pause.fill" : "play.fill")
                        .font(.system(size: 24))
                        .foregroundColor(ColorManager.accentColor)
                        .frame(width: 44, height: 44)
                }
                
                Button(action: {
                    userVideoViewModel.restart()
                    modelVideoViewModel.restart()
                }) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 20))
                        .foregroundColor(ColorManager.textPrimary)
                        .frame(width: 44, height: 44)
                }
            }
            .padding(.bottom, 8)
        }
    }
    
    // 3D overlay comparison view
    private var overlay3DComparisonView: some View {
        VStack(spacing: 16) {
            // 3D View
            CombinedVideo3DView(
                baseViewModel: userVideoViewModel,
                overlayViewModel: modelVideoViewModel
            )
            .frame(height: 400)
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(ColorManager.accentColor, lineWidth: 2)
            )
            .padding(.horizontal, 16)
            
            Text("Blue skeleton: Your technique | Red skeleton: Model technique")
                .font(.caption)
                .foregroundColor(ColorManager.textSecondary)
        }
    }
    
    // Overview Analysis Section - focused on server results
    private var overviewAnalysisSection: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Analysis & Feedback")
                .font(.headline)
                .foregroundColor(ColorManager.textPrimary)
                .padding(.horizontal, 20)
            
            if isAnalyzing {
                // Loading state
                HStack {
                    Spacer()
                    VStack(spacing: 16) {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: ColorManager.accentColor))
                            .scaleEffect(1.5)
                        
                        Text("Analyzing your technique...")
                            .font(.subheadline)
                            .foregroundColor(ColorManager.textPrimary)
                        
                        Text("Comparing with model performance")
                            .font(.caption)
                            .foregroundColor(ColorManager.textSecondary)
                    }
                    .padding(.vertical, 40)
                    Spacer()
                }
            } else if let error = analysisError {
                // Error state
                VStack(alignment: .center, spacing: 16) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 40))
                        .foregroundColor(.orange)
                    
                    Text("Analysis Error")
                        .font(.headline)
                        .foregroundColor(ColorManager.textPrimary)
                    
                    Text(error)
                        .font(.subheadline)
                        .foregroundColor(ColorManager.textSecondary)
                        .multilineTextAlignment(.center)
                    
                    Button(action: {
                        performAnalysis()
                    }) {
                        Text("Retry Analysis")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .background(ColorManager.accentColor)
                            .cornerRadius(8)
                    }
                    .padding(.top, 8)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 30)
                .padding(.horizontal, 20)
                .background(ColorManager.cardBackground.opacity(0.5))
                .cornerRadius(12)
                .padding(.horizontal, 20)
            } else if let result = analysisResult {
                // Success state - show technique score
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Overall Technique Score")
                            .font(.subheadline)
                            .foregroundColor(ColorManager.textPrimary)
                        
                        Text("Based on comparing your form to the model technique")
                            .font(.caption)
                            .foregroundColor(ColorManager.textSecondary)
                    }
                    
                    Spacer()
                    
                    ZStack {
                        Circle()
                            .stroke(ColorManager.accentColor.opacity(0.3), lineWidth: 8)
                            .frame(width: 70, height: 70)
                        
                        Circle()
                            .trim(from: 0, to: CGFloat(result.userScore / 100.0))
                            .stroke(ColorManager.accentColor, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                            .frame(width: 70, height: 70)
                            .rotationEffect(.degrees(-90))
                        
                        Text("\(Int(result.userScore))%")
                            .font(.system(size: 18, weight: .bold))
                            .foregroundColor(ColorManager.textPrimary)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
                .background(ColorManager.cardBackground.opacity(0.5))
                .cornerRadius(12)
                .padding(.horizontal, 20)
                
                // Feedback points based on the analysis
                VStack(alignment: .leading, spacing: 12) {
                    Text("Technique Analysis")
                        .font(.subheadline)
                        .foregroundColor(ColorManager.textPrimary)
                    
                    ForEach(Array(result.userDetails.keys.sorted()), id: \.self) { key in
                        let passed = result.userDetails[key] ?? false
                        let displayName = formatRuleName(key)
                        let description = getDescription(for: key, passed: passed)
                        
                        FeedbackItem(
                            title: displayName,
                            description: description,
                            score: passed ? 95 : 65
                        )
                    }
                }
                .padding(20)
                .background(ColorManager.cardBackground.opacity(0.5))
                .cornerRadius(12)
                .padding(.horizontal, 20)
            } else {
                // Initial state - no analysis yet
                Button(action: {
                    performAnalysis()
                }) {
                    HStack {
                        Image(systemName: "waveform.path")
                            .font(.system(size: 20))
                        Text("Analyze Technique")
                            .font(.headline)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(ColorManager.accentColor)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                }
                .padding(.horizontal, 20)
            }
        }
    }
    
    // Technical Analysis Section
    private var technicalAnalysisSection: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Technical Report")
                .font(.headline)
                .foregroundColor(ColorManager.textPrimary)
                .padding(.horizontal, 20)
            
            if let result = analysisResult {
                // Display comparison details
                VStack(alignment: .leading, spacing: 16) {
                    Text("Comparison with Model")
                        .font(.subheadline)
                        .foregroundColor(ColorManager.textPrimary)
                    
                    HStack(spacing: 20) {
                        // Your score
                        VStack(alignment: .center, spacing: 8) {
                            Text("Your Score")
                                .font(.caption)
                                .foregroundColor(ColorManager.textSecondary)
                            
                            ZStack {
                                Circle()
                                    .stroke(Color.blue.opacity(0.3), lineWidth: 8)
                                    .frame(width: 70, height: 70)
                                
                                Circle()
                                    .trim(from: 0, to: CGFloat(result.userScore / 100.0))
                                    .stroke(Color.blue, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                                    .frame(width: 70, height: 70)
                                    .rotationEffect(.degrees(-90))
                                
                                Text("\(Int(result.userScore))%")
                                    .font(.system(size: 18, weight: .bold))
                                    .foregroundColor(ColorManager.textPrimary)
                            }
                        }
                        .frame(maxWidth: .infinity)
                        
                        // Model score
                        VStack(alignment: .center, spacing: 8) {
                            Text("Model Score")
                                .font(.caption)
                                .foregroundColor(ColorManager.textSecondary)
                            
                            ZStack {
                                Circle()
                                    .stroke(Color.red.opacity(0.3), lineWidth: 8)
                                    .frame(width: 70, height: 70)
                                
                                Circle()
                                    .trim(from: 0, to: CGFloat(result.referenceScore / 100.0))
                                    .stroke(Color.red, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                                    .frame(width: 70, height: 70)
                                    .rotationEffect(.degrees(-90))
                                
                                Text("\(Int(result.referenceScore))%")
                                    .font(.system(size: 18, weight: .bold))
                                    .foregroundColor(ColorManager.textPrimary)
                            }
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .padding(.top, 10)
                    
                    Divider()
                        .padding(.vertical, 10)
                    
                    // Similarity analysis
                    Text("Technical Elements")
                        .font(.subheadline)
                        .foregroundColor(ColorManager.textPrimary)
                    
                    // Column headers
                    HStack {
                        Text("Technique Element")
                            .font(.system(size: 14))
                            .foregroundColor(ColorManager.textSecondary)
                            .frame(width: 180, alignment: .leading)
                        
                        Spacer()
                        
                        Text("Your Performance")
                            .font(.system(size: 14))
                            .foregroundColor(ColorManager.textSecondary)
                            .frame(width: 140, alignment: .trailing)
                    }
                    .padding(.bottom, 8)
                    
                    // List of technique elements
                    ForEach(Array(result.userDetails.keys.sorted()), id: \.self) { key in
                        let userPassed = result.userDetails[key] ?? false
                        let referencePassed = result.referenceDetails[key] ?? false
                        let similar = result.similarity[key] ?? false
                        
                        HStack {
                            Text(formatRuleName(key))
                                .font(.system(size: 16, weight: .medium))
                                .foregroundColor(ColorManager.textPrimary)
                                .frame(width: 180, alignment: .leading)
                                .lineLimit(1)
                            
                            Spacer()
                            
                            HStack(spacing: 8) {
                                Image(systemName: userPassed ? "checkmark.circle.fill" : "xmark.circle.fill")
                                    .foregroundColor(userPassed ? .green : .orange)
                                
                                Text(userPassed ? "Correct" : "Needs Work")
                                    .font(.system(size: 16))
                                    .foregroundColor(userPassed ? .green : .orange)
                            }
                            .frame(width: 140, alignment: .trailing)
                        }
                        .padding(.vertical, 8)
                        .padding(.horizontal, 4)
                        .background(similar ? Color.clear : ColorManager.cardBackground.opacity(0.3))
                        .cornerRadius(8)
                    }
                }
                .padding(20)
                .background(ColorManager.cardBackground.opacity(0.5))
                .cornerRadius(12)
                .padding(.horizontal, 20)
                
                // Improvement suggestions
                VStack(alignment: .leading, spacing: 12) {
                    Text("Improvement Areas")
                        .font(.subheadline)
                        .foregroundColor(ColorManager.textPrimary)
                    
                    let improvementAreas = result.userDetails.filter { !$0.value }.keys.sorted()
                    
                    if improvementAreas.isEmpty {
                        Text("All technical elements performed correctly! Great job!")
                            .font(.system(size: 15))
                            .foregroundColor(.green)
                            .padding(.vertical, 8)
                    } else {
                        ForEach(improvementAreas, id: \.self) { key in
                            Text("• \(getDescription(for: key, passed: false))")
                                .font(.system(size: 15))
                                .foregroundColor(ColorManager.textPrimary)
                                .padding(.vertical, 4)
                        }
                    }
                }
                .padding(20)
                .background(ColorManager.cardBackground.opacity(0.5))
                .cornerRadius(12)
                .padding(.horizontal, 20)
            } else if isAnalyzing {
                // Loading state for technical tab
                HStack {
                    Spacer()
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: ColorManager.accentColor))
                    Spacer()
                }
                .padding(.vertical, 50)
            } else {
                // Prompt to analyze
                HStack {
                    Spacer()
                    VStack(spacing: 16) {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 24))
                            .foregroundColor(ColorManager.textSecondary)
                        
                        Text("Please analyze your technique first")
                            .font(.subheadline)
                            .foregroundColor(ColorManager.textSecondary)
                    }
                    Spacer()
                }
                .padding(.vertical, 50)
            }
        }
    }
    
    // MARK: - Analysis Service Methods
    
    // Perform the analysis with server
    private func performAnalysis() {
        // Clear any previous results and cancel ongoing operations
        analysisResult = nil
        analysisError = nil
        isAnalyzing = true
        cancellables.forEach { $0.cancel() } // Cancel previous Combine subscriptions
        cancellables.removeAll()

        // 1. Clear previously accumulated poses from both view models
        userVideoViewModel.clearAccumulatedPoses()
        modelVideoViewModel.clearAccumulatedPoses()

        // 2. Restart both videos. The restart() method internally calls play().
        userVideoViewModel.restart()
        modelVideoViewModel.restart()
        
        print("performAnalysis: Called restart() on both ViewModels. Waiting for them to become ready.")

        // 3. Wait for both VideoPlayerViewModels to signal they are ready.
        Publishers.CombineLatest(userVideoViewModel.$isVideoReady, modelVideoViewModel.$isVideoReady)
            .filter { userReady, modelReady in
                print("Readiness check: UserVM isReady=\(userReady), ModelVM isReady=\(modelReady)")
                return userReady && modelReady
            }
            .first()
            .receive(on: DispatchQueue.main)
            .sink(receiveCompletion: { completion in // No [weak self]
                if case .failure(let error) = completion {
                    print("Error while waiting for ViewModels to become ready: \(error)")
                    // self still refers to the struct instance here
                    self.isAnalyzing = false
                    self.analysisError = "Failed to prepare videos for analysis. Please try again."
                }
            }, receiveValue: { (userIsReady, modelIsReady) in // No [weak self]
                // self still refers to the struct instance here
                print("Both ViewModels are now ready (User: \(userIsReady), Model: \(modelIsReady)). Proceeding with frame accumulation delay.")
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                    print("Frame accumulation delay ended. Pausing videos prior to analysis.")
                    self.userVideoViewModel.pause()
                    self.modelVideoViewModel.pause()

                    guard self.isAnalyzing else {
                        print("Analysis was cancelled or superseded during accumulation delay.")
                        return
                    }

                    print("Initiating technique comparison with the analysis service.")
                    let service = TechniqueAnalysisService()
                    service.compareTechniques(
                        userViewModel: self.userVideoViewModel,
                        modelViewModel: self.modelVideoViewModel
                    )
                    .sink(
                        receiveCompletion: { serviceCompletionStatus in // No [weak self]
                            // self still refers to the struct instance here
                            self.isAnalyzing = false

                            if case let .failure(error) = serviceCompletionStatus {
                                self.analysisError = error.localizedDescription
                                print("Technique analysis service returned an error: \(error.localizedDescription)")
                            }
                        },
                        receiveValue: { result in // No [weak self]
                            // self still refers to the struct instance here
                            self.analysisResult = result
                            print("Technique analysis successful: User score \(result.userScore)%, Reference score \(result.referenceScore)%")
                        }
                    )
                    .store(in: &self.cancellables) // self.cancellables is fine
                }
            })
            .store(in: &cancellables) // cancellables is a @State var, so self.cancellables is fine
    }
    
    // Helper functions to format rule names and descriptions
    private func formatRuleName(_ rule: String) -> String {
        let names: [String: String] = [
            "shoulder_abduction": "Shoulder Position",
            "elbow_flexion": "Elbow Flexion",
            "elbow_lower": "Elbow Height",
            "foot_direction_aligned": "Foot Alignment",
            "proximal_to_distal_sequence": "Power Sequence",
            "hip_forward_shift": "Hip Forward Movement",
            "trunk_rotation_completed": "Trunk Rotation"
        ]
        return names[rule] ?? rule.replacingOccurrences(of: "_", with: " ").capitalized
    }
    
    private func getDescription(for rule: String, passed: Bool) -> String {
        let descriptions: [String: (pass: String, fail: String)] = [
            "shoulder_abduction": (
                pass: "Great shoulder abduction angle between 60-90°",
                fail: "Improve shoulder abduction angle (aim for 60-90°)"
            ),
            "elbow_flexion": (
                pass: "Good elbow flexion, less than 90°",
                fail: "Increase elbow flexion, aim for angle less than 90°"
            ),
            "elbow_lower": (
                pass: "Correct elbow position, racket arm lower than non-racket arm",
                fail: "Lower your racket arm elbow below your non-racket arm"
            ),
            "foot_direction_aligned": (
                pass: "Proper foot alignment provides good stability",
                fail: "Align your feet in the same direction for better balance"
            ),
            "proximal_to_distal_sequence": (
                pass: "Excellent power generation sequence",
                fail: "Improve your swing sequence: shoulder→elbow→wrist"
            ),
            "hip_forward_shift": (
                pass: "Good hip forward movement for power transfer",
                fail: "Shift your hips forward during the swing for more power"
            ),
            "trunk_rotation_completed": (
                pass: "Full trunk rotation generates maximum power",
                fail: "Complete your trunk rotation for improved power generation"
            )
        ]
        
        if let description = descriptions[rule] {
            return passed ? description.pass : description.fail
        }
        return passed ? "Technique element performed correctly" : "This technique element needs improvement"
    }
}

// MARK: - Supporting Structures

// Feedback item component
struct FeedbackItem: View {
    let title: String
    let description: String
    let score: Int
    
    var body: some View {
        HStack(alignment: .center, spacing: 16) {
            // Score circle
            ZStack {
                Circle()
                    .fill(scoreColor.opacity(0.2))
                    .frame(width: 40, height: 40)
                
                Text("\(score)%")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(scoreColor)
            }
            
            // Feedback text
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline)
                    .foregroundColor(ColorManager.textPrimary)
                
                Text(description)
                    .font(.caption)
                    .foregroundColor(ColorManager.textSecondary)
            }
        }
    }
    
    // Color based on the score
    private var scoreColor: Color {
        if score >= 90 {
            return .green
        } else if score >= 75 {
            return .yellow
        } else if score >= 60 {
            return .orange
        } else {
            return .red
        }
    }
}

// Data structure for results
struct ComparisonResult: Codable {
    let userScore: Double
    let referenceScore: Double
    let similarity: [String: Bool]
    let userDetails: [String: Bool]
    let referenceDetails: [String: Bool]
    
    enum CodingKeys: String, CodingKey {
        case userScore = "user_score"
        case referenceScore = "reference_score"
        case similarity
        case userDetails = "user_details"
        case referenceDetails = "reference_details"
    }
}
