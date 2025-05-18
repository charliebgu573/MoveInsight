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
    @State private var loadingMessage: String? = nil

    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        ZStack {
            ColorManager.background.ignoresSafeArea()
            
            VStack(spacing: 24) {
                // Header
                VStack(spacing: 12) {
                    Text(isComparison ? 
                         LocalizedStringKey("Upload Comparison Video") : 
                         LocalizedStringKey(String(format: NSLocalizedString("Upload %@ Video", comment: ""), technique.name)))
                        .font(.title2)
                        .foregroundColor(ColorManager.textPrimary)
                        .padding(.top, 16)
                    
                    Text(LocalizedStringKey(String(format: NSLocalizedString("Select a video of yourself performing the %@ technique.", comment: ""), technique.name)))
                        .font(.subheadline)
                        .foregroundColor(ColorManager.textSecondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                
                Spacer()
                
                if let message = loadingMessage {
                    VStack {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: ColorManager.accentColor))
                        Text(LocalizedStringKey(message))
                            .foregroundColor(ColorManager.textPrimary)
                            .padding(.top)
                    }
                } else {
                    // Upload button
                    Button(action: {
                        checkPermissionThenPick()
                    }) {
                        VStack(spacing: 16) {
                            Image(systemName: "video.badge.plus")
                                .font(.system(size: 40))
                                .foregroundColor(ColorManager.accentColor)
                            
                            Text(LocalizedStringKey("Select Video from Library"))
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
                    
                    Text(LocalizedStringKey("For best results, ensure your entire body is visible, and you are performing the technique from start to finish."))
                        .font(.caption)
                        .foregroundColor(ColorManager.textSecondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 32)
                }
                
                Spacer()
                
                // Bottom buttons
                Button(LocalizedStringKey("Cancel")) {
                    onVideoSelected(nil)
                    presentationMode.wrappedValue.dismiss()
                }
                .padding()
                .foregroundColor(ColorManager.textSecondary)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 16)
        }
        .photosPicker(
            isPresented: $showPhotoPicker,
            selection: $selectedItem,
            matching: .videos
        )
        .onChange(of: selectedItem) { newItem in
            guard let item = newItem else { return }
            loadingMessage = NSLocalizedString("Loading Video...", comment: "")
            
            // Load the video file URL
            item.loadTransferable(type: VideoItem.self) { result in
                DispatchQueue.main.async {
                    self.loadingMessage = nil 
                    switch result {
                    case .success(let videoItem?):
                        print("Video selected: \(videoItem.url)")
                        onVideoSelected(videoItem.url)
                    case .success(nil):
                        print("Video selection failed: No item returned.")
                        onVideoSelected(nil)
                    case .failure(let error):
                        print("Video selection failed with error: \(error.localizedDescription)")
                        onVideoSelected(nil)
                    }
                }
            }
        }
        .alert(LocalizedStringKey("Photo Library Access Required"), isPresented: $showPermissionAlert) {
            Button(LocalizedStringKey("Go to Settings")) {
                UIApplication.shared.open(URL(string: UIApplication.openSettingsURLString)!)
            }
            Button(LocalizedStringKey("Cancel"), role: .cancel) { }
        }
    }
    
    // MARK: - Permissions
    private func checkPermissionThenPick() {
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        switch status {
        case .authorized, .limited:
            showPhotoPicker = true
        case .notDetermined:
            PHPhotoLibrary.requestAuthorization(for: .readWrite) { newStatus in
                DispatchQueue.main.async {
                    if newStatus == .authorized || newStatus == .limited {
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
}