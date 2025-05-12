import SwiftUI
import Combine

// MARK: - Publisher Extension
extension AnyCancellable {
    func cancel(after interval: TimeInterval) {
        DispatchQueue.main.asyncAfter(deadline: .now() + interval) {
            self.cancel()
        }
    }
}

// MARK: - View Extensions
extension View {
    // Apply a conditional modifier
    @ViewBuilder func applyIf<Content: View>(_ condition: Bool, content: (Self) -> Content) -> some View {
        if condition {
            content(self)
        } else {
            self
        }
    }
    
    // Add a shake effect to a view
    func shake(amount: CGFloat = 5, shakesPerUnit: CGFloat = 3, animationDuration: CGFloat = 0.7, isShaking: Bool = true) -> some View {
        self.modifier(ShakeEffect(amount: amount, shakesPerUnit: shakesPerUnit, animationDuration: animationDuration, isShaking: isShaking))
    }
}

// MARK: - Navigation Extensions
extension View {
    // Create a navigation link that's programmatically triggered
    func navigationLinkWithDestination<Destination: View>(isActive: Binding<Bool>, @ViewBuilder destination: @escaping () -> Destination) -> some View {
        ZStack {
            self
            
            NavigationLink(
                destination: destination(),
                isActive: isActive
            ) {
                EmptyView()
            }
            .hidden()
        }
    }
}

// MARK: - Shake Effect Modifier
struct ShakeEffect: ViewModifier {
    var amount: CGFloat = 5
    var shakesPerUnit: CGFloat = 3
    var animationDuration: CGFloat = 0.7
    var isShaking: Bool = true
    
    func body(content: Content) -> some View {
        content
            .offset(x: isShaking ? amount * sin(shakesPerUnit * .pi * animationDuration) : 0)
            .animation(
                isShaking ?
                    Animation.easeInOut(duration: animationDuration)
                    .repeatForever(autoreverses: true) :
                    .default,
                value: isShaking
            )
    }
}

// MARK: - Dynamic Height Modifier
struct DynamicHeightModifier: ViewModifier {
    @Binding var height: CGFloat
    
    func body(content: Content) -> some View {
        content
            .background(
                GeometryReader { geometry -> Color in
                    DispatchQueue.main.async {
                        self.height = geometry.size.height
                    }
                    return Color.clear
                }
            )
    }
}
