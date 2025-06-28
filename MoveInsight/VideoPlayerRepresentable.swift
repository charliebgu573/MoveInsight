import SwiftUI
import AVKit

// MARK: - Video Player Representable (UIViewRepresentable)
struct VideoPlayerRepresentable: UIViewRepresentable {
    let player: AVPlayer
    @Binding var videoRect: CGRect // Binding to pass videoRect back out

    // Create the custom UIView subclass
    func makeUIView(context: Context) -> PlayerUIView {
        print("Making PlayerUIView")
        let view = PlayerUIView(player: player)
        // Set the callback to update the binding
        view.onVideoRectChange = { rect in
            DispatchQueue.main.async {
                if self.videoRect != rect {
                    self.videoRect = rect
                }
            }
        }
        return view
    }

    // Update the UIView
    func updateUIView(_ uiView: PlayerUIView, context: Context) {
        if uiView.playerLayer.player !== player {
            print("Updating player instance in PlayerUIView")
            uiView.playerLayer.player = player
        }
        uiView.onVideoRectChange = { rect in
            DispatchQueue.main.async {
                if self.videoRect != rect {
                    self.videoRect = rect
                }
            }
        }
        uiView.playerLayer.videoGravity = .resizeAspect
    }
    
    // Clean up resources if needed
    static func dismantleUIView(_ uiView: PlayerUIView, coordinator: ()) {
        print("Dismantling PlayerUIView")
        uiView.playerLayer.player = nil
        uiView.onVideoRectChange = nil // Clear callback
    }
}

// MARK: - Custom UIView for AVPlayerLayer
class PlayerUIView: UIView {
    // Callback closure to report videoRect changes
    var onVideoRectChange: ((CGRect) -> Void)?
    private var lastKnownVideoRect: CGRect = .zero // Store last rect to avoid redundant callbacks

    // Override the layerClass property to specify AVPlayerLayer
    override static var layerClass: AnyClass {
        AVPlayerLayer.self
    }

    // Convenience accessor for the layer as an AVPlayerLayer
    var playerLayer: AVPlayerLayer {
        return layer as! AVPlayerLayer
    }

    // Initializer to set the player on the layer
    init(player: AVPlayer) {
        super.init(frame: .zero)
        playerLayer.player = player
        playerLayer.videoGravity = .resizeAspect // Ensure video scales correctly
        playerLayer.backgroundColor = UIColor.black.cgColor // Set background color for the layer
        self.backgroundColor = .black // Set background for the view itself
        print("PlayerUIView initialized, player assigned to layer.")
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        // Ensure the player layer's frame always matches the view's bounds
        if playerLayer.frame != self.bounds {
            print("LayoutSubviews: Updating playerLayer frame to \(self.bounds)")
            playerLayer.frame = self.bounds
        }
         
        // Get the current videoRect and report if changed
        let currentVideoRect = playerLayer.videoRect
        if currentVideoRect != lastKnownVideoRect && !currentVideoRect.isInfinite && !currentVideoRect.isNull {
            print("LayoutSubviews: videoRect changed to \(currentVideoRect)")
            lastKnownVideoRect = currentVideoRect
            onVideoRectChange?(currentVideoRect) // Call the callback
        }
    }
}
