// Definition of InputSlice layer of NNUE evaluation function

#ifndef _NNUE_LAYERS_INPUT_SLICE_H_
#define _NNUE_LAYERS_INPUT_SLICE_H_

#if defined(EVAL_NNUE)

#include "../nnue_common.h"

namespace Eval {

namespace NNUE {

namespace Layers {

// input layer
template <IndexType OutputDimensions, IndexType Offset = 0>
class InputSlice {
 public:
  // need to maintain alignment
  static_assert(Offset %kMaxSimdWidth == 0, "");

  // output type
  using OutputType = TransformedFeatureType;

  // output dimensionality
  static constexpr IndexType kOutputDimensions = OutputDimensions;

  // Size of the forward propagation buffer used from the input layer to this layer
  static constexpr std::size_t kBufferSize = 0;

  // Hash value embedded in the evaluation function file
  static constexpr std::uint32_t GetHashValue() {
	  auto hash_value = 0xEC42E90Du;
    hash_value ^= kOutputDimensions ^ Offset << 10;
    return hash_value;
  }

  // A string that represents the structure from the input layer to this layer
  static std::string GetStructureString() {
    return "InputSlice[" + std::to_string(kOutputDimensions) + "(" +
        std::to_string(Offset) + ":" +
        std::to_string(Offset + kOutputDimensions) + ")]";
  }

  // read parameters
  static bool ReadParameters(std::istream& /*stream*/) {
    return true;
  }

  // write parameters
  static bool WriteParameters(std::ostream& /*stream*/)
  {
    return true;
  }

  // forward propagation
  static const OutputType* Propagate(
      const TransformedFeatureType* transformed_features,
      char* /*buffer*/)
  {
    return transformed_features + Offset;
  }
};

} // namespace Layers

} // namespace NNUE

} // namespace Eval

#endif // defined(EVAL_NNUE)

#endif