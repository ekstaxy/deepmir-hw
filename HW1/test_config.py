#!/usr/bin/env python3
"""
Test script for configuration system.
Run this to verify that all configurations load and validate correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs import load_experiment_config, validate_config, print_config

def test_configurations():
    """Test loading and validation of all configurations."""

    print("Testing Configuration System")
    print("=" * 50)

    # Test configurations to load
    test_configs = [
        ("traditional_ml", "baseline"),
        ("deep_learning", "baseline")
    ]

    success_count = 0
    total_count = len(test_configs)

    for experiment_type, experiment_name in test_configs:
        try:
            print(f"\nTesting {experiment_type}/{experiment_name}...")

            # Load configuration
            config = load_experiment_config(experiment_type, experiment_name)

            # Validate configuration
            validate_config(config)

            # Print basic info
            print(f"✓ Successfully loaded and validated {experiment_type}/{experiment_name}")
            print(f"  - Experiment type: {config.experiment.type}")
            print(f"  - Description: {config.experiment.description}")
            print(f"  - Number of classes: {config.dataset.num_classes}")
            print(f"  - Sample rate: {config.audio.sample_rate}")

            if experiment_type == "traditional_ml":
                print(f"  - Features to extract: {len(config.features.extract_features)}")
            elif experiment_type == "deep_learning":
                print(f"  - Model type: {config.model.name}")
                print(f"  - Input type: {config.model.input.type}")
                print(f"  - Batch size: {config.training.batch_size}")

            success_count += 1

        except Exception as e:
            print(f"✗ Failed to load {experiment_type}/{experiment_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*50}")
    print(f"Configuration Test Results: {success_count}/{total_count} passed")

    if success_count == total_count:
        print("🎉 All configurations loaded successfully!")
        return True
    else:
        print("❌ Some configurations failed to load")
        return False

def demonstrate_config_usage():
    """Demonstrate how to use the configuration system."""

    print("\n" + "="*50)
    print("Configuration Usage Examples")
    print("="*50)

    # Load a configuration
    config = load_experiment_config("deep_learning", "baseline")

    # Access configuration values
    print("\nAccessing configuration values:")
    print(f"Project name: {config.project.name}")
    print(f"Dataset: {config.dataset.name}")
    print(f"Number of classes: {config.dataset.num_classes}")
    print(f"Model architecture: {config.model.name}")
    print(f"Learning rate: {config.training.learning_rate}")

    # Modify configuration
    print("\nModifying configuration:")
    original_lr = config.training.learning_rate
    config.training.learning_rate = 0.0001
    print(f"Changed learning rate from {original_lr} to {config.training.learning_rate}")

    # Access nested configurations
    print("\nAccessing nested configurations:")
    conv_layers = config.model.cnn.conv_layers
    print(f"Number of CNN layers: {len(conv_layers)}")
    print(f"First layer filters: {conv_layers[0].filters}")
    print(f"Last layer filters: {conv_layers[-1].filters}")

if __name__ == "__main__":
    # Test configurations
    success = test_configurations()

    if success:
        # Demonstrate usage
        demonstrate_config_usage()
        print("\n🎉 Configuration system is ready to use!")
    else:
        print("\n❌ Please fix configuration errors before proceeding")
        sys.exit(1)