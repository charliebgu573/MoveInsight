import SwiftUI
import PhotosUI
import AVKit

struct UploadTabView: View {
    @State private var selectedVideoType: VideoType?
    @State private var showPhotoPicker = false
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedVideoURL: URL?
    @State private var showPermissionAlert = false
    @State private var uploadProgress = 0.0
    @State private var videoPlayerViewModel: VideoPlayerViewModel?
    @State private var currentState: ProcessState = .selectingOptions

    enum VideoType {
        case match, training
    }

    enum ProcessState {
        case selectingOptions
        case uploading
        case playingVideo
    }

    var body: some View {
        ZStack {
            ColorManager.background.ignoresSafeArea()
            VStack {
                // Title
                Text(LocalizedStringKey("Upload Video"))
                    .font(.title2)
                    .foregroundColor(ColorManager.textPrimary)
                    .padding(.top, 16)

                Spacer()

                switch currentState {
                case .selectingOptions:
                    uploadOptionsSelectionView
                case .uploading:
                    uploadingProgressView
                case .playingVideo:
                    if let viewModel = videoPlayerViewModel {
                        VideoWithPoseView(viewModel: viewModel)
                    } else {
                        VStack {
                            ProgressView().tint(ColorManager.accentColor)
                            Text(LocalizedStringKey("Loading Video..."))
                                .foregroundColor(ColorManager.textPrimary)
                                .padding(.top)
                        }
                    }
                }

                Spacer()
            }
            .padding(.horizontal, 16)
            
            // Dismiss button overlay when video is playing
            if currentState == .playingVideo {
                VStack {
                    HStack {
                        Spacer()
                        Button(action: {
                            dismissVideo()
                        }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.title)
                                .foregroundColor(ColorManager.textPrimary)
                        }
                        .padding()
                    }
                    Spacer()
                }
            }
        }
        .photosPicker(
            isPresented: $showPhotoPicker,
            selection: $selectedItem,
            matching: .videos,
            preferredItemEncoding: .automatic,
            photoLibrary: .shared()
        )
        .onChange(of: selectedItem) { newItem in
            selectedVideoURL = nil
            videoPlayerViewModel = nil

            if let newItem = newItem {
                loadVideo(from: newItem)
            }
        }
        .alert(LocalizedStringKey("Photo Library Access Required"), isPresented: $showPermissionAlert) {
            Button(LocalizedStringKey("Go to Settings")) {
                if let url = URL(string: UIApplication.openSettingsURLString),
                   UIApplication.shared.canOpenURL(url) {
                    UIApplication.shared.open(url)
                }
            }
            Button(LocalizedStringKey("Cancel"), role: .cancel) {}
        } message: {
            Text(LocalizedStringKey("Permission to access your photo library is required to upload videos. Please grant access in Settings."))
        }
    }

    private var uploadOptionsSelectionView: some View {
        VStack(spacing: 24) {
            UploadButton(title: LocalizedStringKey("Upload Match Video"), iconName: "sportscourt") {
                selectedVideoType = .match
                checkPhotoLibraryPermission()
            }

            UploadButton(title: LocalizedStringKey("Upload Training Video"), iconName: "figure.run") {
                selectedVideoType = .training
                checkPhotoLibraryPermission()
            }
        }
    }

    private var uploadingProgressView: some View {
        VStack(spacing: 20) {
            ProgressView(value: uploadProgress, total: 1.0)
                .progressViewStyle(LinearProgressViewStyle(tint: ColorManager.accentColor))
                .frame(width: 250)

            Text("\(Int(uploadProgress * 100))%")
                .foregroundColor(ColorManager.textPrimary)
                .font(.headline)

            Text(selectedVideoType == .match ?
                 LocalizedStringKey("Match Video") :
                 LocalizedStringKey("Training Video"))
                .foregroundColor(ColorManager.textSecondary)
                .padding(.top, 10)
        }
    }

    // MARK: - Helper Methods
    private func checkPhotoLibraryPermission() {
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
        case .denied, .restricted:
            showPermissionAlert = true
        @unknown default:
            showPermissionAlert = true
        }
    }

    private func loadVideo(from item: PhotosPickerItem) {
        item.loadTransferable(type: VideoItem.self) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let videoItem?):
                    self.selectedVideoURL = videoItem.url
                    self.startSimulatedUpload()
                case .success(nil):
                    print("Warning: Video item loaded successfully but was nil.")
                case .failure(let error):
                    print("Error loading video: \(error)")
                }
            }
        }
    }

    private func startSimulatedUpload() {
        guard selectedVideoURL != nil, selectedVideoType != nil else {
            print("Missing video URL or type for upload.")
            return
        }
        currentState = .uploading
        uploadProgress = 0.0
        let timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { timer in
            DispatchQueue.main.async {
                if uploadProgress < 1.0 {
                    uploadProgress += 0.02
                    uploadProgress = min(uploadProgress, 1.0)
                } else {
                    timer.invalidate()
                    startVideoAnalysis()
                }
            }
        }
        RunLoop.main.add(timer, forMode: .common)
    }

    private func startVideoAnalysis() {
        guard let videoURL = selectedVideoURL else {
            print("Error: Video URL is nil before starting analysis.")
            currentState = .selectingOptions
            return
        }
        videoPlayerViewModel = VideoPlayerViewModel(videoURL: videoURL)
        currentState = .playingVideo
    }
    
    private func dismissVideo() {
        // Reset state to allow user to re-upload a video
        currentState = .selectingOptions
        selectedVideoURL = nil
        videoPlayerViewModel = nil
        uploadProgress = 0.0
    }
}
