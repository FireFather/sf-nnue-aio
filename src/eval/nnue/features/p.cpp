//Definition of input feature P of NNUE evaluation function

#if defined(EVAL_NNUE)

#include "p.h"
#include "index_list.h"

namespace Eval {

namespace NNUE {

namespace Features {

// Get a list of indices with a value of 1 among the features
void P::AppendActiveIndices(
    const Position& pos, const Color perspective, IndexList* active) {
  // do nothing if array size is small to avoid compiler warning
  if constexpr (RawFeatures::kMaxActiveDimensions < kMaxActiveDimensions) return;

  const BonaPiece* pieces = perspective == BLACK ?
      pos.eval_list()->piece_list_fb() :
      pos.eval_list()->piece_list_fw();
  for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
    if (pieces[i] != BONA_PIECE_ZERO) {
      active->push_back(pieces[i]);
    }
  }
}

// Get a list of indices whose values ​​have changed from the previous one in the feature quantity
void P::AppendChangedIndices(
    const Position& pos, const Color perspective,
    IndexList* removed, IndexList* added) {
  const auto& [changed_piece, pieceNo, dirty_num] = pos.state()->dirtyPiece;
  for (int i = 0; i < dirty_num; ++i) {
    if (pieceNo[i] >= PIECE_NUMBER_KING) continue;
    if (changed_piece[i].old_piece.from[perspective] != BONA_PIECE_ZERO) {
      removed->push_back(changed_piece[i].old_piece.from[perspective]);
    }
    if (changed_piece[i].new_piece.from[perspective] != BONA_PIECE_ZERO) {
      added->push_back(changed_piece[i].new_piece.from[perspective]);
    }
  }
}

}  // namespace Features

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)
