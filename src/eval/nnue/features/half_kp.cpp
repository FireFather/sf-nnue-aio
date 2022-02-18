//Definition of input features HalfKP of NNUE evaluation function

#if defined(EVAL_NNUE)

#include "half_kp.h"
#include "index_list.h"

namespace Eval {

namespace NNUE {

namespace Features {

// Find the index of the feature quantity from the ball position and BonaPiece
template <Side AssociatedKing>
IndexType HalfKP<AssociatedKing>::MakeIndex(const Square sq_k, const BonaPiece p) {
  return static_cast<IndexType>(fe_end) * static_cast<IndexType>(sq_k) + p;
}

// Get the piece information
template <Side AssociatedKing>
void HalfKP<AssociatedKing>::GetPieces(
    const Position& pos, const Color perspective,
    BonaPiece** pieces, Square* sq_target_k) {
  *pieces = perspective == BLACK ?
      pos.eval_list()->piece_list_fb() :
      pos.eval_list()->piece_list_fw();
  const PieceNumber target = AssociatedKing == Side::kFriend ?
      static_cast<PieceNumber>(PIECE_NUMBER_KING + perspective) :
      static_cast<PieceNumber>(PIECE_NUMBER_KING + ~perspective);
  *sq_target_k = static_cast<Square>(((*pieces)[target] - f_king) % SQUARE_NB);
}

// Get a list of indices with a value of 1 among the features
template <Side AssociatedKing>
void HalfKP<AssociatedKing>::AppendActiveIndices(
    const Position& pos, const Color perspective, IndexList* active) {
  // do nothing if array size is small to avoid compiler warning
  if constexpr (RawFeatures::kMaxActiveDimensions < kMaxActiveDimensions) return;

  BonaPiece* pieces;
  Square sq_target_k;
  GetPieces(pos, perspective, &pieces, &sq_target_k);
  for (PieceNumber i = PIECE_NUMBER_ZERO; i < PIECE_NUMBER_KING; ++i) {
    if (pieces[i] != BONA_PIECE_ZERO) {
      active->push_back(MakeIndex(sq_target_k, pieces[i]));
    }
  }
}

// Get a list of indices whose values ​​have changed from the previous one in the feature quantity
template <Side AssociatedKing>
void HalfKP<AssociatedKing>::AppendChangedIndices(
    const Position& pos, const Color perspective,
    IndexList* removed, IndexList* added) {
  BonaPiece* pieces;
  Square sq_target_k;
  GetPieces(pos, perspective, &pieces, &sq_target_k);
  const auto& [changed_piece, pieceNo, dirty_num] = pos.state()->dirtyPiece;
  for (int i = 0; i < dirty_num; ++i) {
    if (pieceNo[i] >= PIECE_NUMBER_KING) continue;
    if (const auto old_p = static_cast<BonaPiece>(
	    changed_piece[i].old_piece.from[perspective]); old_p != BONA_PIECE_ZERO) {
      removed->push_back(MakeIndex(sq_target_k, old_p));
    }
    if (const auto new_p = static_cast<BonaPiece>(
	    changed_piece[i].new_piece.from[perspective]); new_p != BONA_PIECE_ZERO) {
      added->push_back(MakeIndex(sq_target_k, new_p));
    }
  }
}

template class HalfKP<Side::kFriend>;
template class HalfKP<Side::kEnemy>;

}  // namespace Features

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)
