//Definition of input feature quantity K of NNUE evaluation function

#if defined(EVAL_NNUE)

#include "enpassant.h"
#include "index_list.h"

namespace Eval::NNUE::Features
{

	// Get a list of indices with a value of 1 among the features
	void EnPassant::AppendActiveIndices(
		const Position& pos, Color perspective, IndexList* active) {
		// do nothing if array size is small to avoid compiler warning
		if (RawFeatures::kMaxActiveDimensions < kMaxActiveDimensions) return;

		auto epSquare = pos.state()->epSquare;
		if (epSquare == SQ_NONE) {
			return;
		}

		if (perspective == BLACK) {
			epSquare = Inv(epSquare);
		}

		auto file = file_of(epSquare);
		active->push_back(file);
	}

	// Get a list of indexes whose values ​​have changed from the previous one among the feature quantities
	void EnPassant::AppendChangedIndices(
		const Position& pos, Color perspective,
		IndexList* removed, IndexList* added) {
		// Not implemented.
		assert(false);
	}
} // namespace Eval

#endif // defined(EVAL_NNUE)