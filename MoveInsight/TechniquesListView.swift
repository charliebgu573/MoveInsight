import SwiftUI
import AVKit

// Define a structure for badminton techniques
struct BadmintonTechnique: Identifiable {
    let id = UUID()
    let name: String
    let description: String
    let iconName: String
    
    // Flag to indicate if we have a model video for this technique
    let hasModelVideo: Bool
}

struct TechniquesListView: View {
    // List of all badminton techniques
    let techniques = [
        BadmintonTechnique(
            name: NSLocalizedString("Backhand Clear", comment: "Technique name"),
            description: NSLocalizedString("A clear shot played with the back of the hand facing forward.", comment: "Technique description"),
            iconName: "arrow.left.arrow.right",
            hasModelVideo: ModelVideoLoader.shared.hasModelVideo(for: "Backhand Clear")
        ),
        BadmintonTechnique(
            name: NSLocalizedString("Underhand Clear", comment: "Technique name"),
            description: NSLocalizedString("A defensive shot played from below waist height, sending the shuttle high to the back of the opponent's court.", comment: "Technique description"),
            iconName: "arrow.up.forward",
            hasModelVideo: ModelVideoLoader.shared.hasModelVideo(for: "Underhand Clear")
        ),
        BadmintonTechnique(
            name: NSLocalizedString("Overhead Clear", comment: "Technique name"),
            description: NSLocalizedString("A powerful shot played from above the head, sending the shuttle to the back of the opponent's court.", comment: "Technique description"),
            iconName: "arrow.down.forward",
            hasModelVideo: ModelVideoLoader.shared.hasModelVideo(for: "Overhead Clear")
        ),
        BadmintonTechnique(
            name: NSLocalizedString("Drop Shot", comment: "Technique name"),
            description: NSLocalizedString("A gentle shot that just clears the net and drops sharply on the other side.", comment: "Technique description"),
            iconName: "arrow.down",
            hasModelVideo: ModelVideoLoader.shared.hasModelVideo(for: "Drop Shot")
        ),
        BadmintonTechnique(
            name: NSLocalizedString("Smash", comment: "Technique name"),
            description: NSLocalizedString("A powerful overhead shot hit steeply downward into the opponent's court.", comment: "Technique description"),
            iconName: "bolt.fill",
            hasModelVideo: ModelVideoLoader.shared.hasModelVideo(for: "Smash")
        ),
        BadmintonTechnique(
            name: NSLocalizedString("Net Shot", comment: "Technique name"),
            description: NSLocalizedString("A soft shot played near the net that just clears it and falls close to the net on the other side.", comment: "Technique description"),
            iconName: "power.dotted",
            hasModelVideo: ModelVideoLoader.shared.hasModelVideo(for: "Net Shot")
        )
    ]
    
    var body: some View {
        ZStack {
            ColorManager.background.ignoresSafeArea()
            
            ScrollView {
                VStack(spacing: 16) {
                    Text(LocalizedStringKey("Badminton Techniques"))
                        .font(.title)
                        .foregroundColor(ColorManager.textPrimary)
                        .padding(.top, 24)
                    
                    Text(LocalizedStringKey("Select a technique to upload and analyze your form"))
                        .font(.subheadline)
                        .foregroundColor(ColorManager.textSecondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                        .padding(.bottom, 12)
                    
                    LazyVStack(spacing: 16) {
                        ForEach(techniques) { technique in
                            NavigationLink(destination: TechniqueDetailView(technique: technique)) {
                                TechniqueCard(technique: technique)
                            }
                            .buttonStyle(PlainButtonStyle())
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.bottom, 24)
                }
            }
        }
        .navigationTitle(LocalizedStringKey("Techniques"))
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct TechniqueCard: View {
    let technique: BadmintonTechnique
    
    var body: some View {
        HStack(spacing: 16) {
            // Technique icon
            ZStack {
                Circle()
                    .fill(ColorManager.accentColor.opacity(0.2))
                    .frame(width: 60, height: 60)
                
                Image(systemName: technique.iconName)
                    .font(.system(size: 24))
                    .foregroundColor(ColorManager.accentColor)
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text(technique.name)
                    .font(.headline)
                    .foregroundColor(ColorManager.textPrimary)
                
                Text(technique.description)
                    .font(.subheadline)
                    .foregroundColor(ColorManager.textSecondary)
                    .lineLimit(2)
            }
            
            Spacer()
            
            Image(systemName: "chevron.right")
                .foregroundColor(ColorManager.textSecondary)
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(ColorManager.cardBackground)
        )
    }
}