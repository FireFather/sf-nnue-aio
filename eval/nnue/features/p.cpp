//Definition of input feature quantity P of NNUE evaluation function

#if defined(EVAL_NNUE)

#include "p.h"
#include "index_list.h"

namespace Eval::NNUE::Features
{

	// Get a list of indices with a value of 1 among the features
	void P::AppendActiveIndices(
		const Position& pos, const Color perspective, IndexList* active) {
		// do nothing if array size is small to avoid compiler warning
		if constexpr (RawFeatures::kMaxActiveDimensions < kMaxActiveDimensions) return;

		const BonaPiece* pieces = perspective == BLACK ?
			pos.eval_list()->piece_list_fb() :
			pos.eval_list()->piece_list_fw();
		for (auto i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
			if (pieces[i] != BONA_PIECE_ZERO) {
				active->push_back(pieces[i]);
			}
		}
	}

	// Get a list of indexes whose values ​​have changed from the previous one among the feature quantities
	void P::AppendChangedIndices(
		const Position& pos, const Color perspective,
		IndexList* removed, IndexList* added) {
		const auto& dp = pos.state()->dirtyPiece;
		for (auto i = 0; i < dp.dirty_num; ++i) {
			if (dp.pieceNo[i] >= PIECE_NUMBER_KING) continue;
			if (dp.changed_piece[i].old_piece.from[perspective] != BONA_PIECE_ZERO) {
				removed->push_back(dp.changed_piece[i].old_piece.from[perspective]);
			}
			if (dp.changed_piece[i].new_piece.from[perspective] != BONA_PIECE_ZERO) {
				added->push_back(dp.changed_piece[i].new_piece.from[perspective]);
			}
		}
	}
} // namespace Eval

#endif // defined(EVAL_NNUE)