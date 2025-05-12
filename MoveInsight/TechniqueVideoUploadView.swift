import SwiftUI
import PhotosUI
import AVKit

struct TechniqueVideoUploadView: View {
    let technique: BadmintonTechnique
    let isComparison: Bool
    let onVideoSelected: (URL?) -> Void
    
    @State private var selectedItem: PhotosPickerItem?
    @State private var showPhotoPicker = false
    @State private var showPermissionAlert = false
    @State private var uploadProgress = 0.0
    
    @Environment(\.presentationMode) var presentationMode
    
    enum UploadState {
        case initial, uploading, processing, complete, error
    }
    
    @State private var state: UploadState = .initial
    
    var body: some View {
        ZStack {
            ColorManager.background.ignoresSafeArea()
            
            VStack(spacing: 24) {
                // Header
                VStack(spacing: 12) {
                    Text(isComparison ? "Upload Comparison Video" : "Upload \(technique.name) Video")
                        .font(.title2)
                        .foregroundColor(ColorManager.textPrimary)
                        .padding(.top, 16)
                    
                    Text("Select a video of yourself performing the \(technique.name) technique")
                        .font(.subheadline)
                        .foregroundColor(ColorManager.textSecondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                
                Spacer()
                
                // State-dependent content
                switch state {
                case .initial:
                    // Upload button
                    Button(action: {
                        checkPermissionThenPick()
                    }) {
                        VStack(spacing: 16) {
                            Image(systemName: "video.badge.plus")
                                .font(.system(size: 40))
                                .foregroundColor(ColorManager.accentColor)
                            
                            Text("Select Video from Library")
                                .font(.headline)
                                .foregroundColor(ColorManager.textPrimary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(48)
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(ColorManager.accentColor, lineWidth: 2)
                                .background(ColorManager.cardBackground.cornerRadius(12))
                        )
                        .padding(.horizontal, 32)
                    }
                    
                    Text("For best results, ensure your entire body is visible, and you are performing the technique from start to finish.")
                        .font(.caption)
                        .foregroundColor(ColorManager.textSecondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 32)
                
                case .uploading:
                    // Upload progress
                    VStack(spacing: 16) {
                        ProgressView(value: uploadProgress)
                            .progressViewStyle(LinearProgressViewStyle(tint: ColorManager.accentColor))
                            .frame(width: 200)
                        
                        Text("Uploading... \(Int(uploadProgress * 100))%")
                            .foregroundColor(ColorManager.textPrimary)
                    }
                    
                case .processing:
                    // Processing animation
                    VStack(spacing: 16) {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: ColorManager.accentColor))
                            .scaleEffect(1.5)
                        
                        Text("Processing video...")
                            .foregroundColor(ColorManager.textPrimary)
                        
                        Text("Analyzing body movements and technique")
                            .font(.caption)
                            .foregroundColor(ColorManager.textSecondary)
                    }
                    
                case .complete:
                    // Success message
                    VStack(spacing: 16) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 60))
                            .foregroundColor(.green)
                        
                        Text("Video Processed Successfully")
                            .font(.headline)
                            .foregroundColor(ColorManager.textPrimary)
                            
                        Text("Continuing in a moment...")
                            .font(.subheadline)
                            .foregroundColor(ColorManager.textSecondary)
                    }
                    
                case .error:
                    // Error message
                    VStack(spacing: 16) {
                        Image(systemName: "exclamationmark.circle.fill")
                            .font(.system(size: 60))
                            .foregroundColor(.red)
                        
                        Text("Error Processing Video")
                            .font(.headline)
                            .foregroundColor(ColorManager.textPrimary)
                        
                        Button("Try Again") {
                            state = .initial
                        }
                        .padding()
                        .background(ColorManager.accentColor)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                    }
                }
                
                Spacer()
                
                // Bottom buttons
                if state == .initial {
                    Button("Cancel") {
                        presentationMode.wrappedValue.dismiss()
                    }
                    .padding()
                    .foregroundColor(ColorManager.textSecondary)
                }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 16)
        }
        .photosPicker(
            isPresented: $showPhotoPicker,
            selection: $selectedItem,
            matching: .videos
        )
        .onChange(of: selectedItem) { new in
            guard let new = new else { return }
            loadVideo(from: new)
        }
        .alert("Photo Library Access Required", isPresented: $showPermissionAlert) {
            Button("Go to Settings") {
                UIApplication.shared.open(URL(string: UIApplication.openSettingsURLString)!)
            }
            Button("Cancel", role: .cancel) { }
        }
    }
    
    // MARK: - Permissions
    private func checkPermissionThenPick() {
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        switch status {
        case .authorized, .limited:
            showPhotoPicker = true
        case .notDetermined:
            PHPhotoLibrary.requestAuthorization(for: .readWrite) { st in
                DispatchQueue.main.async {
                    if st == .authorized || st == .limited {
                        showPhotoPicker = true
                    } else {
                        showPermissionAlert = true
                    }
                }
            }
        default:
            showPermissionAlert = true
        }
    }

    // MARK: - Loading & "upload"
    private func loadVideo(from item: PhotosPickerItem) {
        // reset progress
        uploadProgress = 0
        state = .uploading
        
        item.loadTransferable(type: VideoItem.self) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let vid?):
                    simulateUploadThenProcess(videoURL: vid.url)
                case .success(nil), .failure(_):
                    state = .error
                }
            }
        }
    }
    
    private func simulateUploadThenProcess(videoURL: URL) {
        // Simulate upload progress
        Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { t in
            uploadProgress += 0.02
            if uploadProgress >= 1 {
                t.invalidate()
                // Move to processing state
                state = .processing
                
                // Simulate processing delay
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    // Complete
                    state = .complete
                    
                    // Return the video URL to the caller but don't dismiss sheet yet
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                        onVideoSelected(videoURL)
                    }
                }
            }
        }.fire()
    }
}
