import SwiftUI

// MARK: - Custom Tab Bar
struct CustomTabBar: View {
    @Binding var selectedTab: Int

    var body: some View {
        EvenlySpacedTabBar(selectedTab: $selectedTab)
    }
}

// MARK: - Evenly Spaced Tab Bar
struct EvenlySpacedTabBar: View {
    @Binding var selectedTab: Int
    
    var body: some View {
        HStack {
            Spacer()
            
            // Home button
            TabBarButtonEvenly(iconName: "house.fill", isSelected: selectedTab == 0) {
                selectedTab = 0
            }
            
            Spacer()
            
            // Training button
            TabBarButtonEvenly(iconName: "figure.run", isSelected: selectedTab == 1) {
                selectedTab = 1
            }
            
            Spacer()
            
            // Plus (upload) button – uses dynamic color for plus icon.
            Button(action: {
                // In this updated design, the Upload tab is directly rendered.
                selectedTab = 2
            }) {
                ZStack {
                    Circle()
                        .fill(ColorManager.accentColor)
                        .frame(width: 44, height: 44)
                    
                    Image(systemName: "plus")
                        .font(.system(size: 18, weight: .bold))
                        .foregroundColor(ColorManager.uploadPlusButtonColor)
                }
            }
            
            Spacer()
            
            // Videos button
            TabBarButtonEvenly(iconName: "play.rectangle.fill", isSelected: selectedTab == 3) {
                selectedTab = 3
            }
            
            Spacer()
            
            // Messages button
            TabBarButtonEvenly(iconName: "message.fill", isSelected: selectedTab == 4) {
                selectedTab = 4
            }
            
            Spacer()
        }
        .padding(.vertical, 10)
        .background(ColorManager.background)
        .overlay(
            Rectangle()
                .frame(height: 1)
                .foregroundColor(ColorManager.textSecondary.opacity(0.3)),
            alignment: .top
        )
    }
}

// MARK: - Tab Bar Button for Evenly Spaced Layout
struct TabBarButtonEvenly: View {
    let iconName: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: iconName)
                .font(.system(size: 20))
                .foregroundColor(isSelected ? ColorManager.accentColor : ColorManager.textPrimary)
        }
    }
}

// MARK: - Legacy Tab Bar Button Component (kept for backward compatibility)
struct TabBarButton: View {
    let iconName: String
    let title: LocalizedStringKey
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 3) {
                Image(systemName: iconName)
                    .font(.system(size: 18))
                    .foregroundColor(isSelected ? ColorManager.accentColor : ColorManager.textPrimary)

                Text(title)
                    .font(.system(size: 11))
                    .foregroundColor(isSelected ? ColorManager.accentColor : ColorManager.textPrimary)
            }
            .frame(maxWidth: .infinity)
        }
    }
}
