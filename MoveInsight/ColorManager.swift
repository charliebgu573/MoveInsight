import SwiftUI

struct ColorManager {
    /// Brand purple color used in MoveInsight logo (#5C2D91)
    static let accentColor = Color(hex: "5C2D91")
    
    // Dynamic background: black in dark mode, white in light mode.
    static var background: Color {
        Color(UIColor { traitCollection in
            traitCollection.userInterfaceStyle == .dark ? UIColor.black : UIColor.white
        })
    }
    
    // Primary text color: white in dark mode, black in light mode.
    static var textPrimary: Color {
        Color(UIColor { traitCollection in
            traitCollection.userInterfaceStyle == .dark ? UIColor.white : UIColor.black
        })
    }
    
    // Secondary text color: light gray in dark mode, dark gray in light mode.
    static var textSecondary: Color {
        Color(UIColor { traitCollection in
            traitCollection.userInterfaceStyle == .dark ? UIColor.lightGray : UIColor.darkGray
        })
    }
    
    // Card background: a subtle gray that adapts to light/dark mode.
    static var cardBackground: Color {
        Color(UIColor { traitCollection in
            if traitCollection.userInterfaceStyle == .dark {
                return UIColor.systemGray6
            } else {
                return UIColor.systemGray5
            }
        })
    }
    
    /// Plus button color: white in light mode, black in dark mode.
    static var uploadPlusButtonColor: Color {
        Color(UIColor { traitCollection in
            traitCollection.userInterfaceStyle == .dark ? UIColor.black : UIColor.white
        })
    }
}

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17,
                                 (int >> 4 & 0xF) * 17,
                                 (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255,
                           int >> 16,
                           int >> 8 & 0xFF,
                           int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24,
                           int >> 16 & 0xFF,
                           int >> 8 & 0xFF,
                           int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0) // fallback: opaque black
        }

        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}
