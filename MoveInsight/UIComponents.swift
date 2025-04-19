import SwiftUI

// MARK: - Custom Upload Button Style
struct UploadButton: View {
    let title: LocalizedStringKey
    let iconName: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: iconName)
                    .font(.system(size: 22))
                Text(title)
                    .font(.headline)
            }
            .foregroundColor(.white) // Always white text
            .frame(maxWidth: .infinity)
            .padding()
            .background(ColorManager.accentColor)
            .cornerRadius(12)
        }
        .padding(.horizontal)
    }
}

// MARK: - Padding Constant
// Define standard padding to use consistently
extension CGFloat {
    static let standard: CGFloat = 16.0
}
