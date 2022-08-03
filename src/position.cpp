/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2020 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <cassert>
#include <cstddef> // For offsetof()
#include <cstring> // For std::memset, std::memcmp
#include <iomanip>
#include <sstream>

#include "bitboard.h"
#include "misc.h"
#include "movegen.h"
#include "position.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"
#include "syzygy/tbprobe.h"

using std::string;

namespace Zobrist {

  Key psq[PIECE_NB][SQUARE_NB];
  Key enpassant[FILE_NB];
  Key castling[CASTLING_RIGHT_NB];
  Key side, noPawns;
}

namespace {

const string PieceToChar(" PNBRQK  pnbrqk");

constexpr Piece Pieces[] = { W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
                             B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING };
} // namespace


/// operator<<(Position) returns an ASCII representation of the position

std::ostream& operator<<(std::ostream& os, const Position& pos) {

  os << "\n +---+---+---+---+---+---+---+---+\n";

  for (Rank r = RANK_8; r >= RANK_1; --r)
  {
      for (File f = FILE_A; f <= FILE_H; ++f)
          os << " | " << PieceToChar[pos.piece_on(make_square(f, r))];

      os << " | " << 1 + r << "\n +---+---+---+---+---+---+---+---+\n";
  }

  os << "   a   b   c   d   e   f   g   h\n"
     << "\nFen: " << pos.fen() << "\nKey: " << std::hex << std::uppercase
     << std::setfill('0') << std::setw(16) << pos.key()
     << std::setfill(' ') << std::dec << "\nCheckers: ";

  for (Bitboard b = pos.checkers(); b; )
      os << UCI::square(pop_lsb(&b)) << " ";

  if (    Tablebases::MaxCardinality >= popcount(pos.pieces())
      && !pos.can_castle(ANY_CASTLING))
  {
      StateInfo st;
      Position p;
      p.set(pos.fen(), pos.is_chess960(), &st, pos.this_thread());
      Tablebases::ProbeState s1, s2;
      const Tablebases::WDLScore wdl = probe_wdl(p, &s1);
      const int dtz = probe_dtz(p, &s2);
      os << "\nTablebases WDL: " << std::setw(4) << wdl << " (" << s1 << ")"
         << "\nTablebases DTZ: " << std::setw(4) << dtz << " (" << s2 << ")";
  }

  return os;
}


// Marcel van Kervinck's cuckoo algorithm for fast detection of "upcoming repetition"
// situations. Description of the algorithm in the following paper:
// https://marcelk.net/2013-04-06/paper/upcoming-rep-v2.pdf

// First and second hash functions for indexing the cuckoo tables
inline int H1(const Key h) { return h & 0x1fff; }
inline int H2(const Key h) { return h >> 16 & 0x1fff; }

// Cuckoo tables with Zobrist hashes of valid reversible moves, and the moves themselves
Key cuckoo[8192];
Move cuckooMove[8192];


/// Position::init() initializes at startup the various arrays used to compute hash keys

void Position::init() {

  PRNG rng(1070372);

  for (const Piece pc : Pieces)
      for (Square s = SQ_A1; s <= SQ_H8; ++s)
          Zobrist::psq[pc][s] = rng.rand<Key>();

  for (File f = FILE_A; f <= FILE_H; ++f)
      Zobrist::enpassant[f] = rng.rand<Key>();

  for (int cr = NO_CASTLING; cr <= ANY_CASTLING; ++cr)
      Zobrist::castling[cr] = rng.rand<Key>();

  Zobrist::side = rng.rand<Key>();
  Zobrist::noPawns = rng.rand<Key>();

  // Prepare the cuckoo tables
  std::memset(cuckoo, 0, sizeof cuckoo);
  std::memset(cuckooMove, 0, sizeof cuckooMove);
  int count = 0;
  for (const Piece pc : Pieces)
      for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1)
          for (auto s2 = static_cast<Square>(s1 + 1); s2 <= SQ_H8; ++s2)
              if (type_of(pc) != PAWN && attacks_bb(type_of(pc), s1, 0) & s2)
              {
                  Move move = make_move(s1, s2);
                  Key key = Zobrist::psq[pc][s1] ^ Zobrist::psq[pc][s2] ^ Zobrist::side;
                  int i = H1(key);
                  while (true)
                  {
                      std::swap(cuckoo[i], key);
                      std::swap(cuckooMove[i], move);
                      if (move == MOVE_NONE) // Arrived at empty slot?
                          break;
                      i = i == H1(key) ? H2(key) : H1(key); // Push victim to alternative slot
                  }
                  count++;
             }
  assert(count == 3668);
}


/// Position::set() initializes the position object with the given FEN string.
/// This function is not very robust - make sure that input FENs are correct,
/// this is assumed to be the responsibility of the GUI.

Position& Position::set(const string& fenStr, const bool isChess960, StateInfo* si, Thread* th) {
/*
   A FEN string defines a particular position using only the ASCII character set.

   A FEN string contains six fields separated by a space. The fields are:

   1) Piece placement (from white's perspective). Each rank is described, starting
      with rank 8 and ending with rank 1. Within each rank, the contents of each
      square are described from file A through file H. Following the Standard
      Algebraic Notation (SAN), each piece is identified by a single letter taken
      from the standard English names. White pieces are designated using upper-case
      letters ("PNBRQK") whilst Black uses lowercase ("pnbrqk"). Blank squares are
      noted using digits 1 through 8 (the number of blank squares), and "/"
      separates ranks.

   2) Active color. "w" means white moves next, "b" means black.

   3) Castling availability. If neither side can castle, this is "-". Otherwise,
      this has one or more letters: "K" (White can castle kingside), "Q" (White
      can castle queenside), "k" (Black can castle kingside), and/or "q" (Black
      can castle queenside).

   4) En passant target square (in algebraic notation). If there's no en passant
      target square, this is "-". If a pawn has just made a 2-square move, this
      is the position "behind" the pawn. Following X-FEN standard, this is recorded only
      if there is a pawn in position to make an en passant capture, and if there really
      is a pawn that might have advanced two squares.

   5) Halfmove clock. This is the number of halfmoves since the last pawn advance
      or capture. This is used to determine if a draw can be claimed under the
      fifty-move rule.

   6) Fullmove number. The number of the full move. It starts at 1, and is
      incremented after Black's move.
*/

unsigned char row;
unsigned char token;
size_t idx;
  Square sq = SQ_A8;
  std::istringstream ss(fenStr);

  std::memset(this, 0, sizeof(Position));
  std::memset(si, 0, sizeof(StateInfo));
  std::fill_n(&pieceList[0][0], sizeof pieceList / sizeof(Square), SQ_NONE);
  st = si;

#if defined(EVAL_NNUE)
  // clear evalList. It is cleared when memset is cleared to zero above...
  evalList.clear();

  // In updating the PieceList, we have to set which piece is where,
  // A counter of how much each piece has been used
  PieceNumber next_piece_number = PIECE_NUMBER_ZERO;
#endif  // defined(EVAL_NNUE)

  ss >> std::noskipws;

  // 1. Piece placement
  while (ss >> token && !isspace(token))
  {
      if (isdigit(token))
          sq += (token - '0') * EAST; // Advance the given number of files

      else if (token == '/')
          sq += 2 * SOUTH;

      else if ((idx = PieceToChar.find(token)) != string::npos)
      {
	      const auto pc = static_cast<Piece>(idx);
          put_piece(pc, sq);

#if defined(EVAL_NNUE)
	      const PieceNumber piece_no =
            idx == W_KING ?PIECE_NUMBER_WKING : //
            idx == B_KING ?PIECE_NUMBER_BKING : // back ball
            next_piece_number++; // otherwise
          evalList.put_piece(piece_no, sq, pc); // Place the pc piece in the sq box
#endif  // defined(EVAL_NNUE)

          ++sq;
      }
  }

  // 2. Active color
  ss >> token;
  sideToMove = token == 'w' ? WHITE : BLACK;
  ss >> token;

  // 3. Castling availability. Compatible with 3 standards: Normal FEN standard,
  // Shredder-FEN that uses the letters of the columns on which the rooks began
  // the game instead of KQkq and also X-FEN standard that, in case of Chess960,
  // if an inner rook is associated with the castling right, the castling tag is
  // replaced by the file letter of the involved rook, as for the Shredder-FEN.
  while (ss >> token && !isspace(token))
  {
      Square rsq;
      const Color c = islower(token) ? BLACK : WHITE;
      const Piece rook = make_piece(c, ROOK);

      token = static_cast<char>(toupper(token));

      if (token == 'K')
          for (rsq = relative_square(c, SQ_H1); piece_on(rsq) != rook; --rsq) {}

      else if (token == 'Q')
          for (rsq = relative_square(c, SQ_A1); piece_on(rsq) != rook; ++rsq) {}

      else if (token >= 'A' && token <= 'H')
          rsq = make_square(static_cast<File>(token - 'A'), relative_rank(c, RANK_1));

      else
          continue;

      set_castling_right(c, rsq);
  }

  // 4. En passant square.
  // Ignore if square is invalid or not on side to move relative rank 6.
  bool enpassant = false;

  if (unsigned char col; ss >> col && (col >= 'a' && col <= 'h')
      && (ss >> row && row == (sideToMove == WHITE ? '6' : '3')))
  {
      st->epSquare = make_square(static_cast<File>(col - 'a'), static_cast<Rank>(row - '1'));

      // En passant square will be considered only if
      // a) side to move have a pawn threatening epSquare
      // b) there is an enemy pawn in front of epSquare
      // c) there is no piece on epSquare or behind epSquare
      enpassant = pawn_attacks_bb(~sideToMove, st->epSquare) & pieces(sideToMove, PAWN)
               && pieces(~sideToMove, PAWN) & st->epSquare + pawn_push(~sideToMove)
               && !(pieces() & (st->epSquare | st->epSquare + pawn_push(sideToMove)));
  }

  if (!enpassant)
      st->epSquare = SQ_NONE;

  // 5-6. Halfmove clock and fullmove number
  ss >> std::skipws >> st->rule50 >> gamePly;

  // Convert from fullmove starting from 1 to gamePly starting from 0,
  // handle also common incorrect FEN with fullmove = 0.
  gamePly = std::max(2 * (gamePly - 1), 0) + (sideToMove == BLACK);

  chess960 = isChess960;
  thisThread = th;
  set_state(st);

  assert(pos_is_ok());
#if defined(EVAL_NNUE)
  assert(evalList.is_valid(*this));
#endif  // defined(EVAL_NNUE)

  return *this;
}


/// Position::set_castling_right() is a helper function used to set castling
/// rights given the corresponding color and the rook starting square.

void Position::set_castling_right(const Color c, const Square rfrom) {
	const Square kfrom = square<KING>(c);
	const CastlingRights cr = c & (kfrom < rfrom ? KING_SIDE: QUEEN_SIDE);

  st->castlingRights |= cr;
  castlingRightsMask[kfrom] |= cr;
  castlingRightsMask[rfrom] |= cr;
  castlingRookSquare[cr] = rfrom;

	const Square kto = relative_square(c, cr & KING_SIDE ? SQ_G1 : SQ_C1);
	const Square rto = relative_square(c, cr & KING_SIDE ? SQ_F1 : SQ_D1);

  castlingPath[cr] =   (between_bb(rfrom, rto) | between_bb(kfrom, kto) | rto | kto)
                    & ~(kfrom | rfrom);
}


/// Position::set_check_info() sets king attacks to detect if a move gives check

void Position::set_check_info(StateInfo* si) const {

  si->blockersForKing[WHITE] = slider_blockers(pieces(BLACK), square<KING>(WHITE), si->pinners[BLACK]);
  si->blockersForKing[BLACK] = slider_blockers(pieces(WHITE), square<KING>(BLACK), si->pinners[WHITE]);

  const Square ksq = square<KING>(~sideToMove);

  si->checkSquares[PAWN]   = pawn_attacks_bb(~sideToMove, ksq);
  si->checkSquares[KNIGHT] = attacks_bb<KNIGHT>(ksq);
  si->checkSquares[BISHOP] = attacks_bb<BISHOP>(ksq, pieces());
  si->checkSquares[ROOK]   = attacks_bb<ROOK>(ksq, pieces());
  si->checkSquares[QUEEN]  = si->checkSquares[BISHOP] | si->checkSquares[ROOK];
  si->checkSquares[KING]   = 0;
}


/// Position::set_state() computes the hash keys of the position, and other
/// data that once computed is updated incrementally as moves are made.
/// The function is only used when a new position is set up, and to verify
/// the correctness of the StateInfo data when running in debug mode.

void Position::set_state(StateInfo* si) const {

  si->key = si->materialKey = 0;
  si->pawnKey = Zobrist::noPawns;
  si->nonPawnMaterial[WHITE] = si->nonPawnMaterial[BLACK] = VALUE_ZERO;
  si->checkersBB = attackers_to(square<KING>(sideToMove)) & pieces(~sideToMove);

  set_check_info(si);

  for (Bitboard b = pieces(); b; )
  {
	  const Square s = pop_lsb(&b);
	  const Piece pc = piece_on(s);
      si->key ^= Zobrist::psq[pc][s];

      if (type_of(pc) == PAWN)
          si->pawnKey ^= Zobrist::psq[pc][s];

      else if (type_of(pc) != KING)
          si->nonPawnMaterial[color_of(pc)] += PieceValue[MG][pc];
  }

  if (si->epSquare != SQ_NONE)
      si->key ^= Zobrist::enpassant[file_of(si->epSquare)];

  if (sideToMove == BLACK)
      si->key ^= Zobrist::side;

  si->key ^= Zobrist::castling[si->castlingRights];

  for (const Piece pc : Pieces)
      for (int cnt = 0; cnt < pieceCount[pc]; ++cnt)
          si->materialKey ^= Zobrist::psq[pc][cnt];
}


/// Position::set() is an overload to initialize the position object with
/// the given endgame code string like "KBPKN". It is mainly a helper to
/// get the material key out of an endgame code.

Position& Position::set(const string& code, const Color c, StateInfo* si) {

  assert(code[0] == 'K');

  string sides[] = { code.substr(code.find('K', 1)),      // Weak
                     code.substr(0, std::min(code.find('v'), code.find('K', 1))) }; // Strong

  assert(sides[0].length() > 0 && sides[0].length() < 8);
  assert(sides[1].length() > 0 && sides[1].length() < 8);

  std::transform(sides[c].begin(), sides[c].end(), sides[c].begin(), tolower);

  const string fenStr = "8/" + sides[0] + static_cast<char>(8 - sides[0].length() + '0') + "/8/8/8/8/"
                       + sides[1] + static_cast<char>(8 - sides[1].length() + '0') + "/8 w - - 0 10";

  return set(fenStr, false, si, nullptr);
}


/// Position::fen() returns a FEN representation of the position. In case of
/// Chess960 the Shredder-FEN notation is used. This is mainly a debugging function.

string Position::fen() const
{

  int emptyCnt;
  std::ostringstream ss;

  for (Rank r = RANK_8; r >= RANK_1; --r)
  {
      for (File f = FILE_A; f <= FILE_H; ++f)
      {
          for (emptyCnt = 0; f <= FILE_H && empty(make_square(f, r)); ++f)
              ++emptyCnt;

          if (emptyCnt)
              ss << emptyCnt;

          if (f <= FILE_H)
              ss << PieceToChar[piece_on(make_square(f, r))];
      }

      if (r > RANK_1)
          ss << '/';
  }

  ss << (sideToMove == WHITE ? " w " : " b ");

  if (can_castle(WHITE_OO))
      ss << (chess960 ? static_cast<char>('A' + file_of(castling_rook_square(WHITE_OO))) : 'K');

  if (can_castle(WHITE_OOO))
      ss << (chess960 ? static_cast<char>('A' + file_of(castling_rook_square(WHITE_OOO))) : 'Q');

  if (can_castle(BLACK_OO))
      ss << (chess960 ? static_cast<char>('a' + file_of(castling_rook_square(BLACK_OO))) : 'k');

  if (can_castle(BLACK_OOO))
      ss << (chess960 ? static_cast<char>('a' + file_of(castling_rook_square(BLACK_OOO))) : 'q');

  if (!can_castle(ANY_CASTLING))
      ss << '-';

  ss << (ep_square() == SQ_NONE ? " - " : " " + UCI::square(ep_square()) + " ")
     << st->rule50 << " " << 1 + (gamePly - (sideToMove == BLACK)) / 2;

  return ss.str();
}


/// Position::slider_blockers() returns a bitboard of all the pieces (both colors)
/// that are blocking attacks on the square 's' from 'sliders'. A piece blocks a
/// slider if removing that piece from the board would result in a position where
/// square 's' is attacked. For example, a king-attack blocking piece can be either
/// a pinned or a discovered check piece, according if its color is the opposite
/// or the same of the color of the slider.

Bitboard Position::slider_blockers(const Bitboard sliders, const Square s, Bitboard& pinners) const {

  Bitboard blockers = 0;
  pinners = 0;

  // Snipers are sliders that attack 's' when a piece and other snipers are removed
  Bitboard snipers = (  attacks_bb<  ROOK>(s) & pieces(QUEEN, ROOK)
                      | attacks_bb<BISHOP>(s) & pieces(QUEEN, BISHOP)) & sliders;
  const Bitboard occupancy = pieces() ^ snipers;

  while (snipers)
  {
	  const Square sniperSq = pop_lsb(&snipers);

	  if (const Bitboard b = between_bb(s, sniperSq) & occupancy; b && !more_than_one(b))
    {
        blockers |= b;
        if (b & pieces(color_of(piece_on(s))))
            pinners |= sniperSq;
    }
  }
  return blockers;
}


/// Position::attackers_to() computes a bitboard of all pieces which attack a
/// given square. Slider attacks use the occupied bitboard to indicate occupancy.

Bitboard Position::attackers_to(const Square s, const Bitboard occupied) const {

  return  pawn_attacks_bb(BLACK, s)       & pieces(WHITE, PAWN)
        | pawn_attacks_bb(WHITE, s)       & pieces(BLACK, PAWN)
        | attacks_bb<KNIGHT>(s)           & pieces(KNIGHT)
        | attacks_bb<  ROOK>(s, occupied) & pieces(  ROOK, QUEEN)
        | attacks_bb<BISHOP>(s, occupied) & pieces(BISHOP, QUEEN)
        | attacks_bb<KING>(s)             & pieces(KING);
}


/// Position::legal() tests whether a pseudo-legal move is legal

bool Position::legal(const Move m) const {

  assert(is_ok(m));

  const Color us = sideToMove;
  const Square from = from_sq(m);
  Square to = to_sq(m);

  assert(color_of(moved_piece(m)) == us);
  assert(piece_on(square<KING>(us)) == make_piece(us, KING));

  // En passant captures are a tricky special case. Because they are rather
  // uncommon, we do it simply by testing whether the king is attacked after
  // the move is made.
  if (type_of(m) == ENPASSANT)
  {
	  const Square ksq = square<KING>(us);
	  const Square capsq = to - pawn_push(us);
	  const Bitboard occupied = pieces() ^ from ^ capsq | to;

      assert(to == ep_square());
      assert(moved_piece(m) == make_piece(us, PAWN));
      assert(piece_on(capsq) == make_piece(~us, PAWN));
      assert(piece_on(to) == NO_PIECE);

      return   !(attacks_bb<  ROOK>(ksq, occupied) & pieces(~us, QUEEN, ROOK))
            && !(attacks_bb<BISHOP>(ksq, occupied) & pieces(~us, QUEEN, BISHOP));
  }

  // Castling moves generation does not check if the castling path is clear of
  // enemy attacks, it is delayed at a later time: now!
  if (type_of(m) == CASTLING)
  {
      // After castling, the rook and king final positions are the same in
      // Chess960 as they would be in standard chess.
      to = relative_square(us, to > from ? SQ_G1 : SQ_C1);
      const Direction step = to > from ? WEST : EAST;

      for (Square s = to; s != from; s += step)
          if (attackers_to(s) & pieces(~us))
              return false;

      // In case of Chess960, verify that when moving the castling rook we do
      // not discover some hidden checker.
      // For instance an enemy queen in SQ_A1 when castling rook is in SQ_B1.
      return   !chess960
            || !(attacks_bb<ROOK>(to, pieces() ^ to_sq(m)) & pieces(~us, ROOK, QUEEN));
  }

  // If the moving piece is a king, check whether the destination square is
  // attacked by the opponent.
  if (type_of(piece_on(from)) == KING)
      return !(attackers_to(to) & pieces(~us));

  // A non-king move is legal if and only if it is not pinned or it
  // is moving along the ray towards or away from the king.
  return   !(blockers_for_king(us) & from)
        ||  aligned(from, to, square<KING>(us));
}


/// Position::pseudo_legal() takes a random move and tests whether the move is
/// pseudo legal. It is used to validate moves from TT that can be corrupted
/// due to SMP concurrent access or hash position key aliasing.

bool Position::pseudo_legal(const Move m) const {
	const Color us = sideToMove;
	const Square from = from_sq(m);
	const Square to = to_sq(m);
	const Piece pc = moved_piece(m);

  // Use a slower but simpler function for uncommon cases
  if (type_of(m) != NORMAL)
      return MoveList<LEGAL>(*this).contains(m);

  // Is not a promotion, so promotion piece must be empty
  if (promotion_type(m) - KNIGHT != NO_PIECE_TYPE)
      return false;

  // If the 'from' square is not occupied by a piece belonging to the side to
  // move, the move is obviously not legal.
  if (pc == NO_PIECE || color_of(pc) != us)
      return false;

  // The destination square cannot be occupied by a friendly piece
  if (pieces(us) & to)
      return false;

  // Handle the special case of a pawn move
  if (type_of(pc) == PAWN)
  {
      // We have already handled promotion moves, so destination
      // cannot be on the 8th/1st rank.
      if ((Rank8BB | Rank1BB) & to)
          return false;

      if (   !(pawn_attacks_bb(us, from) & pieces(~us) & to) // Not a capture
          && !(from + pawn_push(us) == to && empty(to))       // Not a single push
          && !(   from + 2 * pawn_push(us) == to              // Not a double push
               && relative_rank(us, from) == RANK_2
               && empty(to)
               && empty(to - pawn_push(us))))
          return false;
  }
  else if (!(attacks_bb(type_of(pc), from, pieces()) & to))
      return false;

  // Evasions generator already takes care to avoid some kind of illegal moves
  // and legal() relies on this. We therefore have to take care that the same
  // kind of moves are filtered out here.
  if (checkers())
  {
      if (type_of(pc) != KING)
      {
          // Double check? In this case a king move is required
          if (more_than_one(checkers()))
              return false;

          // Our move must be a blocking evasion or a capture of the checking piece
          if (!((between_bb(lsb(checkers()), square<KING>(us)) | checkers()) & to))
              return false;
      }
      // In case of king moves under check we have to remove king so as to catch
      // invalid moves like b1a1 when opposite queen is on c1.
      else if (attackers_to(to, pieces() ^ from) & pieces(~us))
          return false;
  }

  return true;
}


/// Position::gives_check() tests whether a pseudo-legal move gives a check

bool Position::gives_check(const Move m) const {

  assert(is_ok(m));
  assert(color_of(moved_piece(m)) == sideToMove);

  const Square from = from_sq(m);
  const Square to = to_sq(m);

  // Is there a direct check?
  if (check_squares(type_of(piece_on(from))) & to)
      return true;

  // Is there a discovered check?
  if (   blockers_for_king(~sideToMove) & from
      && !aligned(from, to, square<KING>(~sideToMove)))
      return true;

  switch (type_of(m))
  {
  case NORMAL:
      return false;

  case PROMOTION:
      return attacks_bb(promotion_type(m), to, pieces() ^ from) & square<KING>(~sideToMove);

  // En passant capture with check? We have already handled the case
  // of direct checks and ordinary discovered check, so the only case we
  // need to handle is the unusual case of a discovered check through
  // the captured pawn.
  case ENPASSANT:
  {
	  const Square capsq = make_square(file_of(to), rank_of(from));
	  const Bitboard b = pieces() ^ from ^ capsq | to;

      return  attacks_bb<  ROOK>(square<KING>(~sideToMove), b) & pieces(sideToMove, QUEEN, ROOK)
            | attacks_bb<BISHOP>(square<KING>(~sideToMove), b) & pieces(sideToMove, QUEEN, BISHOP);
  }
  case CASTLING:
  {
	  const Square kfrom = from;
	  const Square rfrom = to; // Castling is encoded as 'king captures the rook'
	  const Square kto = relative_square(sideToMove, rfrom > kfrom ? SQ_G1 : SQ_C1);
	  const Square rto = relative_square(sideToMove, rfrom > kfrom ? SQ_F1 : SQ_D1);

      return   attacks_bb<ROOK>(rto) & square<KING>(~sideToMove)
            && attacks_bb<ROOK>(rto, pieces() ^ kfrom ^ rfrom | rto | kto) & square<KING>(~sideToMove);
  }
  default:
      assert(false);
      return false;
  }
}


/// Position::do_move() makes a move, and saves all information necessary
/// to a StateInfo object. The move is assumed to be legal. Pseudo-legal
/// moves should be filtered out before this function is called.

void Position::do_move(const Move m, StateInfo& newSt, const bool givesCheck) {

  assert(is_ok(m));
  assert(&newSt != st);

  thisThread->nodes.fetch_add(1, std::memory_order_relaxed);
  Key k = st->key ^ Zobrist::side;

  // Copy some fields of the old state to our new StateInfo object except the
  // ones which are going to be recalculated from scratch anyway and then switch
  // our state pointer to point to the new (ready to be updated) state.
  std::memcpy(&newSt, st, offsetof(StateInfo, key));
  newSt.previous = st;
  st = &newSt;

  // Increment ply counters. In particular, rule50 will be reset to zero later on
  // in case of a capture or a pawn move.
  ++gamePly;
  ++st->rule50;
  ++st->pliesFromNull;

#if defined(EVAL_NNUE)
  st->accumulator.computed_accumulation = false;
  st->accumulator.computed_score = false;
#endif  // defined(EVAL_NNUE)

  const Color us = sideToMove;
  const Color them = ~us;
  const Square from = from_sq(m);
  Square to = to_sq(m);
  const Piece pc = piece_on(from);
  Piece captured = type_of(m) == ENPASSANT ? make_piece(them, PAWN) : piece_on(to);

#if defined(EVAL_NNUE)
  PieceNumber piece_no0 = PIECE_NUMBER_NB;
  PieceNumber piece_no1 = PIECE_NUMBER_NB;
#endif  // defined(EVAL_NNUE)

  assert(color_of(pc) == us);
  assert(captured == NO_PIECE || color_of(captured) == (type_of(m) != CASTLING ? them : us));
  assert(type_of(captured) != KING);

#if defined(EVAL_NNUE)
  auto& dp = st->dirtyPiece;
  dp.dirty_num = 1;
#endif  // defined(EVAL_NNUE)

  if (type_of(m) == CASTLING)
  {
      assert(pc == make_piece(us, KING));
      assert(captured == make_piece(us, ROOK));

      Square rfrom, rto;
      do_castling<true>(us, from, to, rfrom, rto);

      k ^= Zobrist::psq[captured][rfrom] ^ Zobrist::psq[captured][rto];
      captured = NO_PIECE;
  }

  if (captured)
  {
      Square capsq = to;

      // If the captured piece is a pawn, update pawn hash key, otherwise
      // update non-pawn material.
      if (type_of(captured) == PAWN)
      {
          if (type_of(m) == ENPASSANT)
          {
              capsq -= pawn_push(us);

              assert(pc == make_piece(us, PAWN));
              assert(to == st->epSquare);
              assert(relative_rank(us, to) == RANK_6);
              assert(piece_on(to) == NO_PIECE);
              assert(piece_on(capsq) == make_piece(them, PAWN));

#if defined(EVAL_NNUE)
              piece_no1 = piece_no_of(capsq);
#endif  // defined(EVAL_NNUE)

              //board[capsq] = NO_PIECE; // Not done by remove_piece()
#if defined(EVAL_NNUE)
              evalList.piece_no_list_board[capsq] = PIECE_NUMBER_NB;
#endif  // defined(EVAL_NNUE)
          }
          else {
#if defined(EVAL_NNUE)
            piece_no1 = piece_no_of(capsq);
#endif  // defined(EVAL_NNUE)
          }

          st->pawnKey ^= Zobrist::psq[captured][capsq];
      }
      else {
          st->nonPawnMaterial[them] -= PieceValue[MG][captured];

#if defined(EVAL_NNUE)
          piece_no1 = piece_no_of(capsq);
#endif  // defined(EVAL_NNUE)
      }

      // Update board and piece lists
      remove_piece(capsq);

      if (type_of(m) == ENPASSANT)
          board[capsq] = NO_PIECE;

      // Update material hash key and prefetch access to materialTable
      k ^= Zobrist::psq[captured][capsq];
      st->materialKey ^= Zobrist::psq[captured][pieceCount[captured]];
      prefetch(thisThread->materialTable[st->materialKey]);

      // Reset rule 50 counter
      st->rule50 = 0;

#if defined(EVAL_NNUE)
      dp.dirty_num = 2; // 2 pieces moved

      dp.pieceNo[1] = piece_no1;
      dp.changed_piece[1].old_piece = evalList.bona_piece(piece_no1);
      // Do not use Eval::EvalList::put_piece() because the piece is removed
      // from the game, and the corresponding elements of the piece lists
      // needs to be Eval::BONA_PIECE_ZERO.
      evalList.set_piece_on_board(piece_no1, Eval::BONA_PIECE_ZERO, Eval::BONA_PIECE_ZERO, capsq);
      // Set PIECE_NUMBER_NB to piece_no_of_board[capsq] directly because it
      // will not be overritten to pc if the move type is enpassant.
      evalList.piece_no_list_board[capsq] = PIECE_NUMBER_NB;
      dp.changed_piece[1].new_piece = evalList.bona_piece(piece_no1);
#endif  // defined(EVAL_NNUE)
  }

  // Update hash key
  k ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

  // Reset en passant square
  if (st->epSquare != SQ_NONE)
  {
      k ^= Zobrist::enpassant[file_of(st->epSquare)];
      st->epSquare = SQ_NONE;
  }

  // Update castling rights if needed
  if (st->castlingRights && castlingRightsMask[from] | castlingRightsMask[to])
  {
      k ^= Zobrist::castling[st->castlingRights];
      st->castlingRights &= ~(castlingRightsMask[from] | castlingRightsMask[to]);
      k ^= Zobrist::castling[st->castlingRights];
  }

  // Move the piece. The tricky Chess960 castling is handled earlier
  if (type_of(m) != CASTLING) {
#if defined(EVAL_NNUE)
    piece_no0 = piece_no_of(from);
#endif  // defined(EVAL_NNUE)

    move_piece(from, to);

#if defined(EVAL_NNUE)
    dp.pieceNo[0] = piece_no0;
    dp.changed_piece[0].old_piece = evalList.bona_piece(piece_no0);
    evalList.piece_no_list_board[from] = PIECE_NUMBER_NB;
    evalList.put_piece(piece_no0, to, pc);
    dp.changed_piece[0].new_piece = evalList.bona_piece(piece_no0);
#endif  // defined(EVAL_NNUE)
  }

  // If the moving piece is a pawn do some special extra work
  if (type_of(pc) == PAWN)
  {
      // Set en-passant square if the moved pawn can be captured
      if (   (static_cast<int>(to) ^ static_cast<int>(from)) == 16
          && pawn_attacks_bb(us, to - pawn_push(us)) & pieces(them, PAWN))
      {
          st->epSquare = to - pawn_push(us);
          k ^= Zobrist::enpassant[file_of(st->epSquare)];
      }

      else if (type_of(m) == PROMOTION)
      {
	      const Piece promotion = make_piece(us, promotion_type(m));

          assert(relative_rank(us, to) == RANK_8);
          assert(type_of(promotion) >= KNIGHT && type_of(promotion) <= QUEEN);

          remove_piece(to);
          put_piece(promotion, to);

#if defined(EVAL_NNUE)
          piece_no0 = piece_no_of(to);
          //dp.pieceNo[0] = piece_no0;
          //dp.changed_piece[0].old_piece = evalList.bona_piece(piece_no0);
          assert(evalList.piece_no_list_board[from] == PIECE_NUMBER_NB);
          evalList.put_piece(piece_no0, to, promotion);
          dp.changed_piece[0].new_piece = evalList.bona_piece(piece_no0);
#endif  // defined(EVAL_NNUE)

          // Update hash keys
          k ^= Zobrist::psq[pc][to] ^ Zobrist::psq[promotion][to];
          st->pawnKey ^= Zobrist::psq[pc][to];
          st->materialKey ^=  Zobrist::psq[promotion][pieceCount[promotion]-1]
                            ^ Zobrist::psq[pc][pieceCount[pc]];

          // Update material
          st->nonPawnMaterial[us] += PieceValue[MG][promotion];
      }

      // Update pawn hash key
      st->pawnKey ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

      // Reset rule 50 draw counter
      st->rule50 = 0;
  }

  // Set capture piece
  st->capturedPiece = captured;

  // Update the key with the final value
  st->key = k;

  // Calculate checkers bitboard (if move gives check)
  st->checkersBB = givesCheck ? attackers_to(square<KING>(them)) & pieces(us) : 0;

  sideToMove = ~sideToMove;

  // Update king attacks used for fast check detection
  set_check_info(st);

  // Calculate the repetition info. It is the ply distance from the previous
  // occurrence of the same position, negative in the 3-fold case, or zero
  // if the position was not repeated.
  st->repetition = 0;
  if (const int end = std::min(st->rule50, st->pliesFromNull); end >= 4)
  {
	  const StateInfo* stp = st->previous->previous;
      for (int i = 4; i <= end; i += 2)
      {
          stp = stp->previous->previous;
          if (stp->key == st->key)
          {
              st->repetition = stp->repetition ? -i : i;
              break;
          }
      }
  }

  //std::cout << *this << std::endl;

  assert(pos_is_ok());
#if defined(EVAL_NNUE)
  assert(evalList.is_valid(*this));
#endif  // defined(EVAL_NNUE)
}


/// Position::undo_move() unmakes a move. When it returns, the position should
/// be restored to exactly the same state as before the move was made.

void Position::undo_move(const Move m) {

  assert(is_ok(m));

  sideToMove = ~sideToMove;

  const Color us = sideToMove;
  const Square from = from_sq(m);
  Square to = to_sq(m);
  Piece pc = piece_on(to);

  assert(empty(from) || type_of(m) == CASTLING);
  assert(type_of(st->capturedPiece) != KING);

  if (type_of(m) == PROMOTION)
  {
      assert(relative_rank(us, to) == RANK_8);
      assert(type_of(pc) == promotion_type(m));
      assert(type_of(pc) >= KNIGHT && type_of(pc) <= QUEEN);

      remove_piece(to);
      pc = make_piece(us, PAWN);
      put_piece(pc, to);

#if defined(EVAL_NNUE)
      const PieceNumber piece_no0 = st->dirtyPiece.pieceNo[0];
      evalList.put_piece(piece_no0, to, pc);
#endif  // defined(EVAL_NNUE)
  }

  if (type_of(m) == CASTLING)
  {
      Square rfrom, rto;
      do_castling<false>(us, from, to, rfrom, rto);
  }
  else
  {
      
      move_piece(to, from); // Put the piece back at the source square

#if defined(EVAL_NNUE)
      const PieceNumber piece_no0 = st->dirtyPiece.pieceNo[0];
      evalList.put_piece(piece_no0, from, pc);
      evalList.piece_no_list_board[to] = PIECE_NUMBER_NB;
#endif  // defined(EVAL_NNUE)

      if (st->capturedPiece)
      {
          Square capsq = to;

          if (type_of(m) == ENPASSANT)
          {
              capsq -= pawn_push(us);

              assert(type_of(pc) == PAWN);
              assert(to == st->previous->epSquare);
              assert(relative_rank(us, to) == RANK_6);
              assert(piece_on(capsq) == NO_PIECE);
              assert(st->capturedPiece == make_piece(~us, PAWN));
          }

          put_piece(st->capturedPiece, capsq); // Restore the captured piece

#if defined(EVAL_NNUE)
          const PieceNumber piece_no1 = st->dirtyPiece.pieceNo[1];
          assert(evalList.bona_piece(piece_no1).fw == Eval::BONA_PIECE_ZERO);
          assert(evalList.bona_piece(piece_no1).fb == Eval::BONA_PIECE_ZERO);
          evalList.put_piece(piece_no1, capsq, st->capturedPiece);
#endif  // defined(EVAL_NNUE)
      }
  }

  // Finally point our state pointer back to the previous state
  st = st->previous;
  --gamePly;

  assert(pos_is_ok());
#if defined(EVAL_NNUE)
  assert(evalList.is_valid(*this));
#endif  // defined(EVAL_NNUE)
}


/// Position::do_castling() is a helper used to do/undo a castling move. This
/// is a bit tricky in Chess960 where from/to squares can overlap.
template<bool Do>
void Position::do_castling(const Color us, const Square from, Square& to, Square& rfrom, Square& rto) {
#if defined(EVAL_NNUE)
  auto& dp = st->dirtyPiece;
   // Record the moved pieces in StateInfo for difference calculation.
   dp.dirty_num = 2; // 2 pieces moved

  PieceNumber piece_no0 = {};
  PieceNumber piece_no1 = {};

  if (Do) {
    piece_no0 = piece_no_of(from);
    piece_no1 = piece_no_of(to);
  }
#endif  // defined(EVAL_NNUE)

  const bool kingSide = to > from;
  rfrom = to; // Castling is encoded as "king captures friendly rook"
  rto = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
  to = relative_square(us, kingSide ? SQ_G1 : SQ_C1);

#if defined(EVAL_NNUE)
  if (!Do) {
    piece_no0 = piece_no_of(to);
    piece_no1 = piece_no_of(rto);
  }
#endif  // defined(EVAL_NNUE)

  // Remove both pieces first since squares could overlap in Chess960
  remove_piece(Do ? from : to);
  remove_piece(Do ? rfrom : rto);
  board[Do ? from : to] = board[Do ? rfrom : rto] = NO_PIECE; // Since remove_piece doesn't do this for us
  put_piece(make_piece(us, KING), Do ? to : from);
  put_piece(make_piece(us, ROOK), Do ? rto : rfrom);

#if defined(EVAL_NNUE)
  if (Do) {
    dp.pieceNo[0] = piece_no0;
    dp.changed_piece[0].old_piece = evalList.bona_piece(piece_no0);
    evalList.piece_no_list_board[from] = PIECE_NUMBER_NB;
    evalList.put_piece(piece_no0, to, make_piece(us, KING));
    dp.changed_piece[0].new_piece = evalList.bona_piece(piece_no0);

    dp.pieceNo[1] = piece_no1;
    dp.changed_piece[1].old_piece = evalList.bona_piece(piece_no1);
    evalList.piece_no_list_board[rfrom] = PIECE_NUMBER_NB;
    evalList.put_piece(piece_no1, rto, make_piece(us, ROOK));
    dp.changed_piece[1].new_piece = evalList.bona_piece(piece_no1);
  }
  else {
    evalList.piece_no_list_board[to] = PIECE_NUMBER_NB;
    evalList.put_piece(piece_no0, from, make_piece(us, KING));
    evalList.piece_no_list_board[rto] = PIECE_NUMBER_NB;
    evalList.put_piece(piece_no1, rfrom, make_piece(us, ROOK));
  }
#endif  // defined(EVAL_NNUE)
}


/// Position::do(undo)_null_move() is used to do(undo) a "null move": it flips
/// the side to move without executing any move on the board.

void Position::do_null_move(StateInfo& newSt) {

  assert(!checkers());
  assert(&newSt != st);

  std::memcpy(&newSt, st, sizeof(StateInfo));
  newSt.previous = st;
  st = &newSt;

  if (st->epSquare != SQ_NONE)
  {
      st->key ^= Zobrist::enpassant[file_of(st->epSquare)];
      st->epSquare = SQ_NONE;
  }

  st->key ^= Zobrist::side;
  prefetch(TT.first_entry(st->key));

#if defined(EVAL_NNUE)
  st->accumulator.computed_score = false;
#endif

  ++st->rule50;
  st->pliesFromNull = 0;

  sideToMove = ~sideToMove;

  set_check_info(st);

  st->repetition = 0;

  assert(pos_is_ok());
}

void Position::undo_null_move() {

  assert(!checkers());

  st = st->previous;
  sideToMove = ~sideToMove;
}


/// Position::key_after() computes the new hash key after the given move. Needed
/// for speculative prefetch. It doesn't recognize special moves like castling,
/// en-passant and promotions.

Key Position::key_after(const Move m) const {
	const Square from = from_sq(m);
	const Square to = to_sq(m);
	const Piece pc = piece_on(from);
	const Piece captured = piece_on(to);
  Key k = st->key ^ Zobrist::side;

  if (captured)
      k ^= Zobrist::psq[captured][to];

  return k ^ Zobrist::psq[pc][to] ^ Zobrist::psq[pc][from];
}


/// Position::see_ge (Static Exchange Evaluation Greater or Equal) tests if the
/// SEE value of move is greater or equal to the given threshold. We'll use an
/// algorithm similar to alpha-beta pruning with a null window.

bool Position::see_ge(const Move m, const Value threshold) const {

  assert(is_ok(m));

  // Only deal with normal moves, assume others pass a simple see
  if (type_of(m) != NORMAL)
      return VALUE_ZERO >= threshold;

  const Square from = from_sq(m), to = to_sq(m);

  int swap = PieceValue[MG][piece_on(to)] - threshold;
  if (swap < 0)
      return false;

  swap = PieceValue[MG][piece_on(from)] - swap;
  if (swap <= 0)
      return true;

  Bitboard occupied = pieces() ^ from ^ to;
  Color stm = color_of(piece_on(from));
  Bitboard attackers = attackers_to(to, occupied);
  Bitboard stmAttackers, bb;
  int res = 1;

  while (true)
  {
      stm = ~stm;
      attackers &= occupied;

      // If stm has no more attackers then give up: stm loses
      if (!(stmAttackers = attackers & pieces(stm)))
          break;

      // Don't allow pinned pieces to attack (except the king) as long as
      // there are pinners on their original square.
      if (st->pinners[~stm] & occupied)
          stmAttackers &= ~st->blockersForKing[stm];

      if (!stmAttackers)
          break;

      res ^= 1;

      // Locate and remove the next least valuable attacker, and add to
      // the bitboard 'attackers' any X-ray attackers behind it.
      if ((bb = stmAttackers & pieces(PAWN)))
      {
          if ((swap = PawnValueMg - swap) < res)
              break;

          occupied ^= lsb(bb);
          attackers |= attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN);
      }

      else if ((bb = stmAttackers & pieces(KNIGHT)))
      {
          if ((swap = KnightValueMg - swap) < res)
              break;

          occupied ^= lsb(bb);
      }

      else if ((bb = stmAttackers & pieces(BISHOP)))
      {
          if ((swap = BishopValueMg - swap) < res)
              break;

          occupied ^= lsb(bb);
          attackers |= attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN);
      }

      else if ((bb = stmAttackers & pieces(ROOK)))
      {
          if ((swap = RookValueMg - swap) < res)
              break;

          occupied ^= lsb(bb);
          attackers |= attacks_bb<ROOK>(to, occupied) & pieces(ROOK, QUEEN);
      }

      else if ((bb = stmAttackers & pieces(QUEEN)))
      {
          if ((swap = QueenValueMg - swap) < res)
              break;

          occupied ^= lsb(bb);
          attackers |=  attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN)
                      | attacks_bb<ROOK  >(to, occupied) & pieces(ROOK  , QUEEN);
      }

      else // KING
           // If we "capture" with the king but opponent still has attackers,
           // reverse the result.
          return attackers & ~pieces(stm) ? res ^ 1 : res;
  }

  return static_cast<bool>(res);
}


/// Position::is_draw() tests whether the position is drawn by 50-move rule
/// or by repetition. It does not detect stalemates.

bool Position::is_draw(const int ply) const {

  if (st->rule50 > 99 && (!checkers() || MoveList<LEGAL>(*this).size()))
      return true;

  // Return a draw score if a position repeats once earlier but strictly
  // after the root, or repeats twice before or at the root.
  return st->repetition && st->repetition < ply;
}


// Position::has_repeated() tests whether there has been at least one repetition
// of positions since the last capture or pawn move.

bool Position::has_repeated() const {
	const StateInfo* stc = st;
    int end = std::min(st->rule50, st->pliesFromNull);
    while (end-- >= 4)
    {
        if (stc->repetition)
            return true;

        stc = stc->previous;
    }
    return false;
}


/// Position::has_game_cycle() tests if the position has a move which draws by repetition,
/// or an earlier position has a move that directly reaches the current position.

bool Position::has_game_cycle(const int ply) const {

  int j;

  const int end = std::min(st->rule50, st->pliesFromNull);

  if (end < 3)
    return false;

  const Key originalKey = st->key;
  const StateInfo* stp = st->previous;

  for (int i = 3; i <= end; i += 2)
  {
      stp = stp->previous->previous;

      if (const Key moveKey = originalKey ^ stp->key; (j = H1(moveKey), cuckoo[j] == moveKey)
          || (j = H2(moveKey), cuckoo[j] == moveKey))
      {
	      const Move move = cuckooMove[j];
          Square s1 = from_sq(move);

	      if (Square s2 = to_sq(move); !(between_bb(s1, s2) & pieces()))
          {
              if (ply > i)
                  return true;

              // For nodes before or at the root, check that the move is a
              // repetition rather than a move to the current position.
              // In the cuckoo table, both moves Rc1c5 and Rc5c1 are stored in
              // the same location, so we have to select which square to check.
              if (color_of(piece_on(empty(s1) ? s2 : s1)) != side_to_move())
                  continue;

              // For repetitions before or at the root, require one more
              if (stp->repetition)
                  return true;
          }
      }
  }
  return false;
}


/// Position::flip() flips position with the white and black sides reversed. This
/// is only useful for debugging e.g. for finding evaluation symmetry bugs.

void Position::flip() {

  string f, token;
  std::stringstream ss(fen());

  for (Rank r = RANK_8; r >= RANK_1; --r) // Piece placement
  {
      std::getline(ss, token, r > RANK_1 ? '/' : ' ');
      f.insert(0, token + (f.empty() ? " " : "/"));
  }

  ss >> token; // Active color
  f += token == "w" ? "B " : "W "; // Will be lowercased later

  ss >> token; // Castling availability
  f += token + " ";

  std::transform(f.begin(), f.end(), f.begin(),
                 [](const char c) { return static_cast<char>(islower(c) ? toupper(c) : tolower(c)); });

  ss >> token; // En passant square
  f += token == "-" ? token : token.replace(1, 1, token[1] == '3' ? "6" : "3");

  std::getline(ss, token); // Half and full moves
  f += token;

  set(f, is_chess960(), st, this_thread());

  assert(pos_is_ok());
}


/// Position::pos_is_ok() performs some consistency checks for the
/// position object and raises an asserts if something wrong is detected.
/// This is meant to be helpful when debugging.

bool Position::pos_is_ok() const {
	if (   sideToMove != WHITE && sideToMove != BLACK
      || piece_on(square<KING>(WHITE)) != W_KING
      || piece_on(square<KING>(BLACK)) != B_KING
      || ep_square() != SQ_NONE
      && relative_rank(sideToMove, ep_square()) != RANK_6)
      assert(0 && "pos_is_ok: Default");

  if constexpr (constexpr bool Fast = true)
      return true;

  if (   pieceCount[W_KING] != 1
      || pieceCount[B_KING] != 1
      || attackers_to(square<KING>(~sideToMove)) & pieces(sideToMove))
      assert(0 && "pos_is_ok: Kings");

  if (   pieces(PAWN) & (Rank1BB | Rank8BB)
      || pieceCount[W_PAWN] > 8
      || pieceCount[B_PAWN] > 8)
      assert(0 && "pos_is_ok: Pawns");

  if (   pieces(WHITE) & pieces(BLACK)
      || (pieces(WHITE) | pieces(BLACK)) != pieces()
      || popcount(pieces(WHITE)) > 16
      || popcount(pieces(BLACK)) > 16)
      assert(0 && "pos_is_ok: Bitboards");

  for (PieceType p1 = PAWN; p1 <= KING; ++p1)
      for (PieceType p2 = PAWN; p2 <= KING; ++p2)
          if (p1 != p2 && pieces(p1) & pieces(p2))
              assert(0 && "pos_is_ok: Bitboards");

  StateInfo si = *st;
  set_state(&si);
  if (std::memcmp(&si, st, sizeof(StateInfo)))
      assert(0 && "pos_is_ok: State");

  for (Piece pc : Pieces)
  {
      if (   pieceCount[pc] != popcount(pieces(color_of(pc), type_of(pc)))
          || pieceCount[pc] != std::count(board, board + SQUARE_NB, pc))
          assert(0 && "pos_is_ok: Pieces");

      for (int i = 0; i < pieceCount[pc]; ++i)
          if (board[pieceList[pc][i]] != pc || index[pieceList[pc][i]] != i)
              assert(0 && "pos_is_ok: Index");
  }

  for (const Color c : { WHITE, BLACK })
      for (const CastlingRights cr : {c & KING_SIDE, c & QUEEN_SIDE})
      {
          if (!can_castle(cr))
              continue;

          if (   piece_on(castlingRookSquare[cr]) != make_piece(c, ROOK)
              || castlingRightsMask[castlingRookSquare[cr]] != cr
              || (castlingRightsMask[square<KING>(c)] & cr) != cr)
              assert(0 && "pos_is_ok: Castling");
      }

  return true;
}

#if defined(EVAL_NNUE)
PieceNumber Position::piece_no_of(const Square sq) const
{
  assert(piece_on(sq) != NO_PIECE);
  const PieceNumber n = evalList.piece_no_of_board(sq);
  assert(is_ok(n));
  return n;
}
#endif  // defined(EVAL_NNUE)
