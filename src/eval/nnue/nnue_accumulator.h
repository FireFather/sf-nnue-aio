// Class for difference calculation of NNUE evaluation function

#pragma once

#include "nnue_architecture.h"

namespace Eval {

	namespace NNUE {

		// Class that holds the result of affine transformation of input features
		// Keep the evaluation value that is the final output together
		struct alignas(32) Accumulator {
			std::int16_t
				accumulation[2][kRefreshTriggers.size()][kTransformedFeatureDimensions];
			Value score = VALUE_ZERO;
			bool computed_accumulation = false;
			bool computed_score = false;
		};

	}  // namespace NNUE

}  // namespace Eval

