// swift-tools-version: 6.0
// Changed by Hunch, 2026: renamed the package and library to Hunch, dropped the direct Hub dependency.
// Derived from laya-mlx-swift (https://github.com/MstyAI/laya-mlx-swift, Apache-2.0); see NOTICE.

import PackageDescription

let package = Package(
  name: "hunch",
  platforms: [.macOS(.v14)],
  products: [
    .library(name: "Hunch", targets: ["Hunch"])
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
    .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
  ],
  targets: [
    .target(
      name: "Hunch",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXFast", package: "mlx-swift"),
        .product(name: "Tokenizers", package: "swift-transformers"),
      ]
    ),
    .testTarget(
      name: "HunchTests",
      dependencies: ["Hunch"],
      resources: [.copy("Fixtures")]
    ),
  ]
)
