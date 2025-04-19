import SwiftUI
import Vision

// MARK: - Pose Overlay View (SwiftUI)
struct PoseOverlayView: View {
    let poses: [[VNHumanBodyPoseObservation.JointName: CGPoint]]
    let connections: [BodyConnection]
    let videoRect: CGRect

    var body: some View {
        Canvas { context, size in
            guard !videoRect.isEmpty,
                  !videoRect.isInfinite,
                  !videoRect.isNull,
                  videoRect.width > 0,
                  videoRect.height > 0 else {
                return
            }
            
            for bodyParts in poses {
                for connection in connections {
                    guard let fromPointNorm = bodyParts[connection.from],
                          let toPointNorm = bodyParts[connection.to] else { continue }

                    let fromPointView = CGPoint(
                        x: videoRect.origin.x + fromPointNorm.x * videoRect.size.width,
                        y: videoRect.origin.y + fromPointNorm.y * videoRect.size.height
                    )
                    let toPointView = CGPoint(
                        x: videoRect.origin.x + toPointNorm.x * videoRect.size.width,
                        y: videoRect.origin.y + toPointNorm.y * videoRect.size.height
                    )

                    var path = Path()
                    path.move(to: fromPointView)
                    path.addLine(to: toPointView)

                    context.stroke(path, with: .color(ColorManager.accentColor), lineWidth: 3)
                }

                for (_, pointNorm) in bodyParts {
                    let pointView = CGPoint(
                        x: videoRect.origin.x + pointNorm.x * videoRect.size.width,
                        y: videoRect.origin.y + pointNorm.y * videoRect.size.height
                    )

                    let jointRect = CGRect(x: pointView.x - 4, y: pointView.y - 4, width: 8, height: 8)

                    context.fill(Path(ellipseIn: jointRect), with: .color(ColorManager.textPrimary))
                    context.stroke(Path(ellipseIn: jointRect.insetBy(dx: -1, dy: -1)), with: .color(ColorManager.accentColor), lineWidth: 1)
                }
            }
        }
    }
}
