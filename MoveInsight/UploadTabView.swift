import SwiftUI
import PhotosUI
import AVKit

struct UploadTabView: View {
    @State private var navigateToTechniquesList = false
    
    var body: some View {
        NavigationView {
            ZStack {
                ColorManager.background.ignoresSafeArea()
                
                VStack(spacing: 30) {
                    // Header
                    Text(LocalizedStringKey("Upload Video"))
                        .font(.title)
                        .foregroundColor(ColorManager.textPrimary)
                        .padding(.top, 40)
                    
                    Text(LocalizedStringKey("Choose the type of video you want to upload"))
                        .font(.subheadline)
                        .foregroundColor(ColorManager.textSecondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 32)
                    
                    Spacer()
                    
                    // Technique Video Option
                    NavigationLink(destination: TechniquesListView(), isActive: $navigateToTechniquesList) {
                        EmptyView()
                    }
                    
                    Button(action: {
                        navigateToTechniquesList = true
                    }) {
                        UploadOptionCard(
                            title: LocalizedStringKey("Upload Technique Video"),
                            description: LocalizedStringKey("Analyze and compare your badminton techniques with model performers"),
                            icon: "figure.badminton"
                        )
                    }
                    .buttonStyle(ScaleButtonStyle())
                    
                    // Match Video Option
                    Button(action: {
                        // Do nothing for now, as per requirements
                    }) {
                        UploadOptionCard(
                            title: LocalizedStringKey("Upload Match Video"),
                            description: LocalizedStringKey("Upload your match videos for performance analysis"),
                            icon: "sportscourt"
                        )
                    }
                    .buttonStyle(ScaleButtonStyle())
                    
                    Spacer()
                }
                .padding(.horizontal, 20)
            }
            .navigationBarHidden(true)
        }
    }
}

// Card view for upload options
struct UploadOptionCard: View {
    let title: LocalizedStringKey
    let description: LocalizedStringKey
    let icon: String
    
    var body: some View {
        HStack(spacing: 20) {
            // Icon
            ZStack {
                Circle()
                    .fill(ColorManager.accentColor.opacity(0.2))
                    .frame(width: 70, height: 70)
                
                Image(systemName: icon)
                    .font(.system(size: 30))
                    .foregroundColor(ColorManager.accentColor)
            }
            
            // Text content
            VStack(alignment: .leading, spacing: 8) {
                Text(title)
                    .font(.headline)
                    .foregroundColor(ColorManager.textPrimary)
                
                Text(description)
                    .font(.subheadline)
                    .foregroundColor(ColorManager.textSecondary)
                    .lineLimit(2)
            }
            
            Spacer()
            
            // Arrow
            Image(systemName: "chevron.right")
                .foregroundColor(ColorManager.textSecondary)
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(ColorManager.cardBackground)
        )
    }
}

struct ScaleButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.97 : 1.0)
            .animation(.easeInOut(duration: 0.2), value: configuration.isPressed)
    }
}