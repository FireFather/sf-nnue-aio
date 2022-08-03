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

#ifndef TYPES_H_INCLUDED
#define TYPES_H_INCLUDED

/// When compiling with provided Makefile (e.g. for Linux and OSX), configuration
/// is done automatically. To get started type 'make help'.
///
/// When Makefile is not used (e.g. with Microsoft Visual Studio) some switches
/// need to be set manually:
///
/// -DNDEBUG      | Disable debugging mode. Always use this for release.
///
/// -DNO_PREFETCH | Disable use of prefetch asm-instruction. You may need this to
///               | run on some very old machines.
///
/// -DUSE_POPCNT  | Add runtime support for use of popcnt asm-instruction. Works
///               | only in 64-bit mode and requires hardware with popcnt support.
///
/// -DUSE_PEXT    | Add runtime support for use of pext asm-instruction. Works
///               | only in 64-bit mode and requires hardware with pext support.

#include <cassert>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <algorithm>

#if defined(_MSC_VER)
// Disable some silly and noisy warning from MSVC compiler
#pragma warning(disable: 4127) // Conditional expression is constant
#pragma warning(disable: 4146) // Unary minus operator applied to unsigned type
#pragma warning(disable: 4800) // Forcing value to bool 'true' or 'false'
#else
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wparentheses"
#endif

/// Predefined macros hell:
///
/// __GNUC__           Compiler is gcc, Clang or Intel on Linux
/// __INTEL_COMPILER   Compiler is Intel
/// _MSC_VER           Compiler is MSVC or Intel on Windows
/// _WIN32             Building on Windows (any)
/// _WIN64             Building on Windows 64 bit

#if defined(_WIN64) && defined(_MSC_VER) // No Makefile used
#  include <intrin.h> // Microsoft header for _BitScanForward64()
#  define IS_64BIT
#endif

#if defined(USE_POPCNT) && (defined(__INTEL_COMPILER) || defined(_MSC_VER))
#  include <nmmintrin.h> // Intel and Microsoft header for _mm_popcnt_u64()
#endif

#if !defined(NO_PREFETCH) && (defined(__INTEL_COMPILER) || defined(_MSC_VER))
#  include <xmmintrin.h> // Intel and Microsoft header for _mm_prefetch()
#endif

#if defined(USE_PEXT)
#  include <immintrin.h> // Header for _pext_u64() intrinsic
#  define pext(b, m) _pext_u64(b, m)
#else
#  define pext(b, m) 0
#endif

#ifdef USE_AVX2
constexpr bool HasAvx2 = true;
#else
constexpr bool HasAvx2 = false;
#endif

#ifdef USE_POPCNT
constexpr bool HasPopCnt = true;
#else
constexpr bool HasPopCnt = false;
#endif

#ifdef USE_PEXT
constexpr bool HasPext = true;
#else
constexpr bool HasPext = false;
#endif

#ifdef IS_64BIT
constexpr bool Is64Bit = true;
#else
constexpr bool Is64Bit = false;
#endif

typedef uint64_t Key;
typedef uint64_t Bitboard;

constexpr int MAX_MOVES = 256;
constexpr int MAX_PLY   = 246;

/// A move needs 16 bits to be stored
///
/// bit  0- 5: destination square (from 0 to 63)
/// bit  6-11: origin square (from 0 to 63)
/// bit 12-13: promotion piece type - 2 (from KNIGHT-2 to QUEEN-2)
/// bit 14-15: special move flag: promotion (1), en passant (2), castling (3)
/// NOTE: EN-PASSANT bit is set only when a pawn can be captured
///
/// Special cases are MOVE_NONE and MOVE_NULL. We can sneak these in because in
/// any normal move destination square is always different from origin square
/// while MOVE_NONE and MOVE_NULL have the same origin and destination square.

enum Move : int {
  MOVE_NONE,
  MOVE_NULL = 65
};

enum MoveType {
  NORMAL,
  PROMOTION = 1 << 14,
  ENPASSANT = 2 << 14,
  CASTLING  = 3 << 14
};

enum Color {
  WHITE, BLACK, COLOR_NB = 2
};

constexpr Color Colors[2] = { WHITE, BLACK };

enum CastlingRights {
  NO_CASTLING,
  WHITE_OO,
  WHITE_OOO = WHITE_OO << 1,
  BLACK_OO  = WHITE_OO << 2,
  BLACK_OOO = WHITE_OO << 3,

  KING_SIDE      = WHITE_OO  | BLACK_OO,
  QUEEN_SIDE     = WHITE_OOO | BLACK_OOO,
  WHITE_CASTLING = WHITE_OO  | WHITE_OOO,
  BLACK_CASTLING = BLACK_OO  | BLACK_OOO,
  ANY_CASTLING   = WHITE_CASTLING | BLACK_CASTLING,

  CASTLING_RIGHT_NB = 16
};

enum Phase {
  PHASE_ENDGAME,
  PHASE_MIDGAME = 128,
  MG = 0, EG = 1, PHASE_NB = 2
};

enum ScaleFactor {
  SCALE_FACTOR_DRAW    = 0,
  SCALE_FACTOR_NORMAL  = 64,
  SCALE_FACTOR_MAX     = 128,
  SCALE_FACTOR_NONE    = 255
};

enum Bound {
  BOUND_NONE,
  BOUND_UPPER,
  BOUND_LOWER,
  BOUND_EXACT = BOUND_UPPER | BOUND_LOWER
};

enum Value : int {
  VALUE_ZERO      = 0,
  VALUE_DRAW      = 0,
  VALUE_KNOWN_WIN = 10000,
  VALUE_MATE      = 32000,
  VALUE_INFINITE  = 32001,
  VALUE_NONE      = 32002,

  VALUE_TB_WIN_IN_MAX_PLY  =  VALUE_MATE - 2 * MAX_PLY,
  VALUE_TB_LOSS_IN_MAX_PLY = -VALUE_TB_WIN_IN_MAX_PLY,
  VALUE_MATE_IN_MAX_PLY  =  VALUE_MATE - MAX_PLY,
  VALUE_MATED_IN_MAX_PLY = -VALUE_MATE_IN_MAX_PLY,

  PawnValueMg   = 124,   PawnValueEg   = 206,
  KnightValueMg = 781,   KnightValueEg = 854,
  BishopValueMg = 825,   BishopValueEg = 915,
  RookValueMg   = 1276,  RookValueEg   = 1380,
  QueenValueMg  = 2538,  QueenValueEg  = 2682,
  Tempo = 28,

  MidgameLimit  = 15258, EndgameLimit  = 3915,

// Maximum value returned by the evaluation function (I want it to be around 2**14..)
  VALUE_MAX_EVAL = 27000,
};

enum PieceType {
  NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
  ALL_PIECES = 7,
  PIECE_TYPE_NB = 8
};

enum Piece {
  NO_PIECE,
  W_PAWN = 1, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
  B_PAWN = 9, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
  PIECE_NB = 16
};

constexpr Value PieceValue[PHASE_NB][PIECE_NB] = {
  { VALUE_ZERO, PawnValueMg, KnightValueMg, BishopValueMg, RookValueMg, QueenValueMg, VALUE_ZERO, VALUE_ZERO,
    VALUE_ZERO, PawnValueMg, KnightValueMg, BishopValueMg, RookValueMg, QueenValueMg, VALUE_ZERO, VALUE_ZERO },
  { VALUE_ZERO, PawnValueEg, KnightValueEg, BishopValueEg, RookValueEg, QueenValueEg, VALUE_ZERO, VALUE_ZERO,
    VALUE_ZERO, PawnValueEg, KnightValueEg, BishopValueEg, RookValueEg, QueenValueEg, VALUE_ZERO, VALUE_ZERO }
};

typedef int Depth;

enum : int {
  DEPTH_QS_CHECKS     =  0,
  DEPTH_QS_NO_CHECKS  = -1,
  DEPTH_QS_RECAPTURES = -5,

  DEPTH_NONE   = -6,
  DEPTH_OFFSET = DEPTH_NONE
};

enum Square : int {
  SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
  SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
  SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
  SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
  SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
  SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
  SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
  SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
  SQ_NONE,

  SQUARE_ZERO = 0, SQUARE_NB = 64,
  SQUARE_NB_PLUS1 = SQUARE_NB + 1, // If there are no balls, it is treated as having moved to SQUARE_NB, so it may be necessary to secure the array with SQUARE_NB+1, so this constant is used.
};

enum Direction : int {
  NORTH =  8,
  EAST  =  1,
  SOUTH = -NORTH,
  WEST  = -EAST,

  NORTH_EAST = NORTH + EAST,
  SOUTH_EAST = SOUTH + EAST,
  SOUTH_WEST = SOUTH + WEST,
  NORTH_WEST = NORTH + WEST
};

enum File : int {
  FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_NB
};

enum Rank : int {
  RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_NB
};


/// Score enum stores a middlegame and an endgame value in a single integer (enum).
/// The least significant 16 bits are used to store the middlegame value and the
/// upper 16 bits are used to store the endgame value. We have to take care to
/// avoid left-shifting a signed int to avoid undefined behavior.
enum Score : int { SCORE_ZERO };

constexpr Score make_score(const int mg, const int eg) {
  return static_cast<Score>(static_cast<int>(static_cast<unsigned>(eg) << 16) + mg);
}

/// Extracting the signed lower and upper 16 bits is not so trivial because
/// according to the standard a simple cast to short is implementation defined
/// and so is a right shift of a signed integer.
inline Value eg_value(const Score s) {
  union { uint16_t u; int16_t s; } eg = { static_cast<uint16_t>(static_cast<unsigned>(s + 0x8000) >> 16) };
  return static_cast<Value>(eg.s);
}

inline Value mg_value(const Score s) {
  union { uint16_t u; int16_t s; } mg = { static_cast<uint16_t>(static_cast<unsigned>(s)) };
  return static_cast<Value>(mg.s);
}

#define ENABLE_BASE_OPERATORS_ON(T)                                \
constexpr T operator+(T d1, int d2) { return T(int(d1) + d2); } \
constexpr T operator-(T d1, int d2) { return T(int(d1) - d2); } \
constexpr T operator-(T d) { return T(-int(d)); }                  \
inline T& operator+=(T& d1, int d2) { return d1 = d1 + d2; }         \
inline T& operator-=(T& d1, int d2) { return d1 = d1 - d2; }

#define ENABLE_INCR_OPERATORS_ON(T)                                \
inline T& operator++(T& d) { return d = T(int(d) + 1); }           \
inline T& operator--(T& d) { return d = T(int(d) - 1); }

#define ENABLE_FULL_OPERATORS_ON(T)                                \
ENABLE_BASE_OPERATORS_ON(T)                                        \
constexpr T operator*(int i, T d) { return T(i * int(d)); }        \
constexpr T operator*(T d, int i) { return T(int(d) * i); }        \
constexpr T operator/(T d, int i) { return T(int(d) / i); }        \
constexpr int operator/(T d1, T d2) { return int(d1) / int(d2); }  \
inline T& operator*=(T& d, int i) { return d = T(int(d) * i); }    \
inline T& operator/=(T& d, int i) { return d = T(int(d) / i); }

ENABLE_FULL_OPERATORS_ON(Value)
ENABLE_FULL_OPERATORS_ON(Direction)

ENABLE_INCR_OPERATORS_ON(PieceType)
ENABLE_INCR_OPERATORS_ON(Piece)
ENABLE_INCR_OPERATORS_ON(Square)
ENABLE_INCR_OPERATORS_ON(File)
ENABLE_INCR_OPERATORS_ON(Rank)

ENABLE_BASE_OPERATORS_ON(Score)

#undef ENABLE_FULL_OPERATORS_ON
#undef ENABLE_INCR_OPERATORS_ON
#undef ENABLE_BASE_OPERATORS_ON

/// Additional operators to add a Direction to a Square
constexpr Square operator+(const Square s, const Direction d) { return static_cast<Square>(static_cast<int>(s) + static_cast<int>(d)); }
constexpr Square operator-(const Square s, const Direction d) { return static_cast<Square>(static_cast<int>(s) - static_cast<int>(d)); }
inline Square& operator+=(Square& s, const Direction d) { return s = s + d; }
inline Square& operator-=(Square& s, const Direction d) { return s = s - d; }

/// Only declared but not defined. We don't want to multiply two scores due to
/// a very high risk of overflow. So user should explicitly convert to integer.
Score operator*(Score, Score) = delete;

/// Division of a Score must be handled separately for each term
inline Score operator/(const Score s, const int i) {
  return make_score(mg_value(s) / i, eg_value(s) / i);
}

/// Multiplication of a Score by an integer. We check for overflow in debug mode.
inline Score operator*(const Score s, const int i) {
	const auto result = static_cast<Score>(static_cast<int>(s) * i);

  assert(eg_value(result) == (i * eg_value(s)));
  assert(mg_value(result) == (i * mg_value(s)));
  assert((i == 0) || (result / i) == s);

  return result;
}

/// Multiplication of a Score by a boolean
inline Score operator*(const Score s, const bool b) {
  return b ? s : SCORE_ZERO;
}

constexpr Color operator~(const Color c) {
  return static_cast<Color>(c ^ BLACK); // Toggle color
}

constexpr Square flip_rank(const Square s) { // Swap A1 <-> A8
  return static_cast<Square>(s ^ SQ_A8);
}

constexpr Square flip_file(const Square s) { // Swap A1 <-> H1
  return static_cast<Square>(s ^ SQ_H1);
}

constexpr Piece operator~(const Piece pc) {
  return static_cast<Piece>(pc ^ 8); // Swap color of piece B_KNIGHT <-> W_KNIGHT
}

constexpr CastlingRights operator&(const Color c, const CastlingRights cr) {
  return static_cast<CastlingRights>((c == WHITE ? WHITE_CASTLING : BLACK_CASTLING) & cr);
}

constexpr Value mate_in(const int ply) {
  return VALUE_MATE - ply;
}

constexpr Value mated_in(const int ply) {
  return -VALUE_MATE + ply;
}

constexpr Square make_square(const File f, const Rank r) {
  return static_cast<Square>((r << 3) + f);
}

constexpr Piece make_piece(const Color c, const PieceType pt) {
  return static_cast<Piece>((c << 3) + pt);
}

constexpr PieceType type_of(const Piece pc) {
  return static_cast<PieceType>(pc & 7);
}

inline Color color_of(const Piece pc) {
  assert(pc != NO_PIECE);
  return static_cast<Color>(pc >> 3);
}

constexpr bool is_ok(const Square s) {
  return s >= SQ_A1 && s <= SQ_H8;
}

constexpr File file_of(const Square s) {
  return static_cast<File>(s & 7);
}

constexpr Rank rank_of(const Square s) {
  return static_cast<Rank>(s >> 3);
}

constexpr Square relative_square(const Color c, const Square s) {
  return static_cast<Square>(s ^ c * 56);
}

constexpr Rank relative_rank(const Color c, const Rank r) {
  return static_cast<Rank>(r ^ c * 7);
}

constexpr Rank relative_rank(const Color c, const Square s) {
  return relative_rank(c, rank_of(s));
}

constexpr Direction pawn_push(const Color c) {
  return c == WHITE ? NORTH : SOUTH;
}

constexpr Square from_sq(const Move m) {
  return static_cast<Square>(m >> 6 & 0x3F);
}

constexpr Square to_sq(const Move m) {
  return static_cast<Square>(m & 0x3F);
}

constexpr int from_to(const Move m) {
 return m & 0xFFF;
}

constexpr MoveType type_of(const Move m) {
  return static_cast<MoveType>(m & 3 << 14);
}

constexpr PieceType promotion_type(const Move m) {
  return static_cast<PieceType>((m >> 12 & 3) + KNIGHT);
}

constexpr Move make_move(const Square from, const Square to) {
  return static_cast<Move>((from << 6) + to);
}

constexpr Move reverse_move(const Move m) {
  return make_move(to_sq(m), from_sq(m));
}

template<MoveType T>
constexpr Move make(const Square from, const Square to, const PieceType pt = KNIGHT) {
  return static_cast<Move>(T + (pt - KNIGHT << 12) + (from << 6) + to);
}

constexpr bool is_ok(const Move m) {
  return from_sq(m) != to_sq(m); // Catch MOVE_NULL and MOVE_NONE
}

// Return squares when turning the board 180��
constexpr Square Inv(const Square sq) { return static_cast<Square>(SQUARE_NB - 1 - sq); }

// Return squares when mirroring the board
constexpr Square Mir(const Square sq) { return make_square(static_cast<File>(7 - static_cast<int>(file_of(sq))), rank_of(sq)); }

#if defined(EVAL_NNUE) || defined(EVAL_LEARN)
// --------------------
// 		piece box
// --------------------

// A number used to manage the piece list (which piece is where) used in the Position class.
enum PieceNumber : uint8_t
{
	PIECE_NUMBER_PAWN = 0,
	PIECE_NUMBER_KNIGHT = 16,
	PIECE_NUMBER_BISHOP = 20,
	PIECE_NUMBER_ROOK = 24,
	PIECE_NUMBER_QUEEN = 28,
	PIECE_NUMBER_KING = 30,
	PIECE_NUMBER_WKING = 30,
	PIECE_NUMBER_BKING = 31, // Use this if you need the numbers of the first and second balls
	PIECE_NUMBER_ZERO = 0,
	PIECE_NUMBER_NB = 32,
};

inline PieceNumber& operator++(PieceNumber& d) { return d = static_cast<PieceNumber>(static_cast<int8_t>(d) + 1); }
inline PieceNumber operator++(PieceNumber& d, int) {
	const PieceNumber x = d;
  d = static_cast<PieceNumber>(static_cast<int8_t>(d) + 1);
  return x;
}
inline PieceNumber& operator--(PieceNumber& d) { return d = static_cast<PieceNumber>(static_cast<int8_t>(d) - 1); }

// Piece Number integrity check. for assert.
constexpr bool is_ok(const PieceNumber pn) { return pn < PIECE_NUMBER_NB; }
#endif  // defined(EVAL_NNUE) || defined(EVAL_LEARN)

/// Based on a congruential pseudo random number generator
constexpr Key make_key(const uint64_t seed) {
  return seed * 6364136223846793005ULL + 1442695040888963407ULL;
}

#endif // #ifndef TYPES_H_INCLUDED

#include "tune.h" // Global visibility to tuning setup
