﻿// Specialization of NNUE evaluation function learning class template for AffineTransform

#ifndef _NNUE_TRAINER_AFFINE_TRANSFORM_H_
#define _NNUE_TRAINER_AFFINE_TRANSFORM_H_

#if defined(EVAL_LEARN) && defined(EVAL_NNUE)

#include "../../../learn/learn.h"
#include "../layers/affine_transform.h"
#include "trainer.h"

#include <random>

namespace Eval {

namespace NNUE {

// Learning: Affine transformation layer
template <typename PreviousLayer, IndexType OutputDimensions>
class Trainer<Layers::AffineTransform<PreviousLayer, OutputDimensions>> {
	// Type of layer to learn
  using LayerType = Layers::AffineTransform<PreviousLayer, OutputDimensions>;

 public:
  // factory function
  static std::shared_ptr<Trainer> Create(
      LayerType* target_layer, FeatureTransformer* feature_transformer) {
    return std::shared_ptr<Trainer>(
        new Trainer(target_layer, feature_transformer));
  }

  // Set options such as hyperparameters
  void SendMessage(Message* message) {
    previous_layer_trainer_->SendMessage(message);
    if (ReceiveMessage("momentum", message)) {
      momentum_ = static_cast<LearnFloatType>(std::stod(message->value));
    }
    if (ReceiveMessage("learning_rate_scale", message)) {
      learning_rate_scale_ =
          static_cast<LearnFloatType>(std::stod(message->value));
    }
    if (ReceiveMessage("reset", message)) {
      DequantizeParameters();
    }
    if (ReceiveMessage("quantize_parameters", message)) {
      QuantizeParameters();
    }
  }

  // Initialize the parameters with random numbers
  template <typename RNG>
  void Initialize(RNG& rng) {
    previous_layer_trainer_->Initialize(rng);
    if (kIsOutputLayer) {
      // Initialize output layer with 0
      std::fill(std::begin(biases_), std::end(biases_),
                static_cast<LearnFloatType>(0.0));
      std::fill(std::begin(weights_), std::end(weights_),
                static_cast<LearnFloatType>(0.0));
    } else {
      // Assuming that the input distribution is unit-mean 0.5, equal variance,
      // Initialize the output distribution so that each unit has a mean of 0.5 and the same variance as the input
      const double kSigma = 1.0 / std::sqrt(kInputDimensions);
      auto distribution = std::normal_distribution(0.0, kSigma);
      for (IndexType i = 0; i < kOutputDimensions; ++i) {
        double sum = 0.0;
        for (IndexType j = 0; j < kInputDimensions; ++j) {
          const auto weight = static_cast<LearnFloatType>(distribution(rng));
          weights_[kInputDimensions * i + j] = weight;
          sum += weight;
        }
        biases_[i] = static_cast<LearnFloatType>(0.5 - 0.5 * sum);
      }
    }
    QuantizeParameters();
  }

  // forward propagation
  const LearnFloatType* Propagate(const std::vector<Example>& batch) {
    if (output_.size() < kOutputDimensions * batch.size()) {
      output_.resize(kOutputDimensions * batch.size());
      gradients_.resize(kInputDimensions * batch.size());
    }
    batch_size_ = static_cast<IndexType>(batch.size());
    batch_input_ = previous_layer_trainer_->Propagate(batch);
#if defined(USE_BLAS)
    for (IndexType b = 0; b < batch_size_; ++b) {
      const IndexType batch_offset = kOutputDimensions * b;
      cblas_scopy(kOutputDimensions, biases_, 1, &output_[batch_offset], 1);
    }
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                kOutputDimensions, batch_size_, kInputDimensions, 1.0,
                weights_, kInputDimensions,
                batch_input_, kInputDimensions,
                1.0, &output_[0], kOutputDimensions);
#else
    for (IndexType b = 0; b < batch_size_; ++b) {
      const IndexType input_batch_offset = kInputDimensions * b;
      const IndexType output_batch_offset = kOutputDimensions * b;
      for (IndexType i = 0; i < kOutputDimensions; ++i) {
        double sum = biases_[i];
        for (IndexType j = 0; j < kInputDimensions; ++j) {
          const IndexType index = kInputDimensions * i + j;
          sum += weights_[index] * batch_input_[input_batch_offset + j];
        }
        output_[output_batch_offset + i] = static_cast<LearnFloatType>(sum);
      }
    }
#endif
    return output_.data();
  }

  // backpropagation
  void Backpropagate(const LearnFloatType* gradients,
                     LearnFloatType learning_rate) {
    const LearnFloatType local_learning_rate =
        learning_rate * learning_rate_scale_;
#if defined(USE_BLAS)
    // backpropagate
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                kInputDimensions, batch_size_, kOutputDimensions, 1.0,
                weights_, kInputDimensions,
                gradients, kOutputDimensions,
                0.0, &gradients_[0], kInputDimensions);
    // update
    cblas_sscal(kOutputDimensions, momentum_, biases_diff_, 1);
    for (IndexType b = 0; b < batch_size_; ++b) {
      const IndexType batch_offset = kOutputDimensions * b;
      cblas_saxpy(kOutputDimensions, 1.0,
                  &gradients[batch_offset], 1, biases_diff_, 1);
    }
    cblas_saxpy(kOutputDimensions, -local_learning_rate,
                biases_diff_, 1, biases_, 1);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                kOutputDimensions, kInputDimensions, batch_size_, 1.0,
                gradients, kOutputDimensions,
                batch_input_, kInputDimensions,
                momentum_, weights_diff_, kInputDimensions);
    cblas_saxpy(kOutputDimensions * kInputDimensions, -local_learning_rate,
                weights_diff_, 1, weights_, 1);
#else
    // backpropagate
    for (IndexType b = 0; b < batch_size_; ++b) {
      const IndexType input_batch_offset = kInputDimensions * b;
      const IndexType output_batch_offset = kOutputDimensions * b;
      for (IndexType j = 0; j < kInputDimensions; ++j) {
        double sum = 0.0;
        for (IndexType i = 0; i < kOutputDimensions; ++i) {
          const IndexType index = kInputDimensions * i + j;
          sum += weights_[index] * gradients[output_batch_offset + i];
        }
        gradients_[input_batch_offset + j] = static_cast<LearnFloatType>(sum);
      }
    }
    // update
    for (IndexType i = 0; i < kOutputDimensions; ++i) {
      biases_diff_[i] *= momentum_;
    }
    for (IndexType i = 0; i < kOutputDimensions * kInputDimensions; ++i) {
      weights_diff_[i] *= momentum_;
    }
    for (IndexType b = 0; b < batch_size_; ++b) {
      const IndexType input_batch_offset = kInputDimensions * b;
      const IndexType output_batch_offset = kOutputDimensions * b;
      for (IndexType i = 0; i < kOutputDimensions; ++i) {
        biases_diff_[i] += gradients[output_batch_offset + i];
      }
      for (IndexType i = 0; i < kOutputDimensions; ++i) {
        for (IndexType j = 0; j < kInputDimensions; ++j) {
          const IndexType index = kInputDimensions * i + j;
          weights_diff_[index] += gradients[output_batch_offset + i] *
              batch_input_[input_batch_offset + j];
        }
      }
    }
    for (IndexType i = 0; i < kOutputDimensions; ++i) {
      biases_[i] -= local_learning_rate * biases_diff_[i];
    }
    for (IndexType i = 0; i < kOutputDimensions * kInputDimensions; ++i) {
      weights_[i] -= local_learning_rate * weights_diff_[i];
    }
#endif
    previous_layer_trainer_->Backpropagate(gradients_.data(), learning_rate);
  }

 private:
  // constructor
  Trainer(LayerType* target_layer, FeatureTransformer* feature_transformer) :
      batch_size_(0),
      batch_input_(nullptr),
      previous_layer_trainer_(Trainer<PreviousLayer>::Create(
          &target_layer->previous_layer_, feature_transformer)),
      target_layer_(target_layer),
      biases_(),
      weights_(),
      biases_diff_(),
      weights_diff_(),
      momentum_(0.0),
      learning_rate_scale_(1.0) {
    DequantizeParameters();
  }

  // Weight saturation and parameterization
  void QuantizeParameters() {
    for (IndexType i = 0; i < kOutputDimensions * kInputDimensions; ++i) {
      weights_[i] = std::max(-kMaxWeightMagnitude,
                             std::min(+kMaxWeightMagnitude, weights_[i]));
    }
    for (IndexType i = 0; i < kOutputDimensions; ++i) {
      target_layer_->biases_[i] =
          Round<typename LayerType::BiasType>(biases_[i] * kBiasScale);
    }
    for (IndexType i = 0; i < kOutputDimensions; ++i) {
      const auto offset = kInputDimensions * i;
      const auto padded_offset = LayerType::kPaddedInputDimensions * i;
      for (IndexType j = 0; j < kInputDimensions; ++j) {
        target_layer_->weights_[padded_offset + j] =
            Round<typename LayerType::WeightType>(
                weights_[offset + j] * kWeightScale);
      }
    }
  }

  // read parameterized integer
  void DequantizeParameters() {
    for (IndexType i = 0; i < kOutputDimensions; ++i) {
      biases_[i] = static_cast<LearnFloatType>(
          target_layer_->biases_[i] / kBiasScale);
    }
    for (IndexType i = 0; i < kOutputDimensions; ++i) {
      const auto offset = kInputDimensions * i;
      const auto padded_offset = LayerType::kPaddedInputDimensions * i;
      for (IndexType j = 0; j < kInputDimensions; ++j) {
        weights_[offset + j] = static_cast<LearnFloatType>(
            target_layer_->weights_[padded_offset + j] / kWeightScale);
      }
    }
    std::fill(std::begin(biases_diff_), std::end(biases_diff_),
              static_cast<LearnFloatType>(0.0));
    std::fill(std::begin(weights_diff_), std::end(weights_diff_),
              static_cast<LearnFloatType>(0.0));
  }

  // number of input/output dimensions
  static constexpr IndexType kInputDimensions = LayerType::kInputDimensions;
  static constexpr IndexType kOutputDimensions = LayerType::kOutputDimensions;

  // If the output dimensionality is 1, the output layer
  static constexpr bool kIsOutputLayer = kOutputDimensions == 1;

  // Coefficient used for parameterization
  static constexpr LearnFloatType kActivationScale =
      std::numeric_limits<std::int8_t>::max();
  static constexpr LearnFloatType kBiasScale = kIsOutputLayer ?
      kPonanzaConstant * FV_SCALE :
      (1 << kWeightScaleBits) * kActivationScale;
  static constexpr LearnFloatType kWeightScale = kBiasScale / kActivationScale;

  // Upper limit of absolute value of weight used to prevent overflow when parameterizing integers
  static constexpr LearnFloatType kMaxWeightMagnitude =
      std::numeric_limits<typename LayerType::WeightType>::max() / kWeightScale;

  // number of samples in mini-batch
  IndexType batch_size_;

  // Input mini batch
  const LearnFloatType* batch_input_;

  // Trainer of the previous layer
  const std::shared_ptr<Trainer<PreviousLayer>> previous_layer_trainer_;

  // layer to learn
  LayerType* const target_layer_;

  // parameter
  LearnFloatType biases_[kOutputDimensions];
  LearnFloatType weights_[kOutputDimensions * kInputDimensions];

  // Buffer used for updating parameters
  LearnFloatType biases_diff_[kOutputDimensions];
  LearnFloatType weights_diff_[kOutputDimensions * kInputDimensions];

  // Forward propagation buffer
  std::vector<LearnFloatType> output_;

  // buffer for back propagation
  std::vector<LearnFloatType> gradients_;

  // hyper parameter
  LearnFloatType momentum_;
  LearnFloatType learning_rate_scale_;
};

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_LEARN) && defined(EVAL_NNUE)

#endif
