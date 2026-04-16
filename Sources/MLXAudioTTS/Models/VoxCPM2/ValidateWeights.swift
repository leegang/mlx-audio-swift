import Foundation
import HuggingFace
import MLX
import MLXNN
import MLXAudioCore

// Standalone validation helper for VoxCPM2 weight mapping.
// This can be run as a Swift script or XCTest to check key alignment
// against the mlx-community/VoxCPM2-4bit checkpoint.

public func validateVoxCPM2Weights(repoID: String = "mlx-community/VoxCPM2-4bit") async throws {
    guard let repo = Repo.ID(rawValue: repoID) else {
        throw NSError(domain: "ValidateWeights", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid repo ID"])
    }

    // 1. Download config and instantiate model
    let configURL = try await ModelUtils.resolveOrDownloadModel(
        repoID: repo,
        requiredExtension: "json",
        additionalMatchingPatterns: ["config.json"]
    ).appendingPathComponent("config.json")

    let configData = try Data(contentsOf: configURL)
    let config = try JSONDecoder().decode(VoxCPM2Config.self, from: configData)
    let model = try VoxCPM2Model(config: config)

    // 2. Download weights
    let weightsURL = try await ModelUtils.resolveOrDownloadModel(
        repoID: repo,
        requiredExtension: ".safetensors",
        additionalMatchingPatterns: ["model.safetensors"]
    ).appendingPathComponent("model.safetensors")

    let weights = try MLX.loadArrays(url: weightsURL)
    let sanitized = model.sanitize(weights: weights)

    let modelKeys = Set(model.parameters().flattened().map { $0.0 })
    let weightKeys = Set(sanitized.keys)

    let missingInModel = weightKeys.subtracting(modelKeys)
    let missingInWeights = modelKeys.subtracting(weightKeys)

    print("=== VoxCPM2 Weight Validation ===")
    print("Model keys:   \(modelKeys.count)")
    print("Weight keys:  \(weightKeys.count)")

    if !missingInModel.isEmpty {
        print("\n❌ Weight keys not found in model (\(missingInModel.count)):")
        for k in missingInModel.sorted() {
            print("  - \(k)")
        }
    } else {
        print("\n✅ All weight keys matched in model.")
    }

    if !missingInWeights.isEmpty {
        print("\n⚠️ Model keys not in weights (\(missingInWeights.count)):")
        for k in missingInWeights.sorted() {
            print("  - \(k)")
        }
    } else {
        print("\n✅ All model keys matched in weights.")
    }
}
