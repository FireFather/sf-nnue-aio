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
#include <cstring>   // For std::memset
#include <iomanip>
#include <set>
#include <sstream>

#include "bitboard.h"
#include "evaluate.h"
#include "material.h"
#include "pawns.h"
#include "thread.h"
#include "uci.h"
#include "eval/nnue/evaluate_nnue.h"

namespace Trace {

  enum Tracing { NO_TRACE, TRACE };

  enum Term { // The first 8 entries are reserved for PieceType
    MATERIAL = 8, IMBALANCE, MOBILITY, THREAT, PASSED, SPACE, WINNABLE, TOTAL, TERM_NB
  };

  Score scores[TERM_NB][COLOR_NB];

  double to_cp(const Value v) { return static_cast<double>(v) / PawnValueEg; }

  void add(const int idx, const Color c, const Score s) {
    scores[idx][c] = s;
  }

  void add(const int idx, const Score w, const Score b = SCORE_ZERO) {
    scores[idx][WHITE] = w;
    scores[idx][BLACK] = b;
  }

  std::ostream& operator<<(std::ostream& os, const Score s) {
    os << std::setw(5) << to_cp(mg_value(s)) << " "
       << std::setw(5) << to_cp(eg_value(s));
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const Term t) {

    if (t == MATERIAL || t == IMBALANCE || t == WINNABLE || t == TOTAL)
        os << " ----  ----"    << " | " << " ----  ----";
    else
        os << scores[t][WHITE] << " | " << scores[t][BLACK];

    os << " | " << scores[t][WHITE] - scores[t][BLACK] << "\n";
    return os;
  }
}

using namespace Trace;

namespace {

  // Threshold for lazy and space evaluation
  constexpr Value LazyThreshold  = static_cast<Value>(1400);
  constexpr Value SpaceThreshold = static_cast<Value>(12222);

  // KingAttackWeights[PieceType] contains king attack weights by piece type
  constexpr int KingAttackWeights[PIECE_TYPE_NB] = { 0, 0, 81, 52, 44, 10 };

  // SafeCheck[PieceType][single/multiple] contains safe check bonus by piece type,
  // higher if multiple safe checks are possible for that piece type.
  constexpr int SafeCheck[][2] = {
      {}, {}, {792, 1283}, {645, 967}, {1084, 1897}, {772, 1119}
  };

#define S(mg, eg) make_score(mg, eg)

  // MobilityBonus[PieceType-2][attacked] contains bonuses for middle and end game,
  // indexed by piece type and number of attacked squares in the mobility area.
  constexpr Score MobilityBonus[][32] = {
    { S(-62,-81), S(-53,-56), S(-12,-31), S( -4,-16), S(  3,  5), S( 13, 11), // Knight
      S( 22, 17), S( 28, 20), S( 33, 25) },
    { S(-48,-59), S(-20,-23), S( 16, -3), S( 26, 13), S( 38, 24), S( 51, 42), // Bishop
      S( 55, 54), S( 63, 57), S( 63, 65), S( 68, 73), S( 81, 78), S( 81, 86),
      S( 91, 88), S( 98, 97) },
    { S(-60,-78), S(-20,-17), S(  2, 23), S(  3, 39), S(  3, 70), S( 11, 99), // Rook
      S( 22,103), S( 31,121), S( 40,134), S( 40,139), S( 41,158), S( 48,164),
      S( 57,168), S( 57,169), S( 62,172) },
    { S(-30,-48), S(-12,-30), S( -8, -7), S( -9, 19), S( 20, 40), S( 23, 55), // Queen
      S( 23, 59), S( 35, 75), S( 38, 78), S( 53, 96), S( 64, 96), S( 65,100),
      S( 65,121), S( 66,127), S( 67,131), S( 67,133), S( 72,136), S( 72,141),
      S( 77,147), S( 79,150), S( 93,151), S(108,168), S(108,168), S(108,171),
      S(110,182), S(114,182), S(114,192), S(116,219) }
  };

  // KingProtector[knight/bishop] contains penalty for each distance unit to own king
  constexpr Score KingProtector[] = { S(8, 9), S(6, 9) };

  // Outpost[knight/bishop] contains bonuses for each knight or bishop occupying a
  // pawn protected square on rank 4 to 6 which is also safe from a pawn attack.
  constexpr Score Outpost[] = { S(56, 36), S(30, 23) };

  // PassedRank[Rank] contains a bonus according to the rank of a passed pawn
  constexpr Score PassedRank[RANK_NB] = {
    S(0, 0), S(10, 28), S(17, 33), S(15, 41), S(62, 72), S(168, 177), S(276, 260)
  };

  // RookOnFile[semiopen/open] contains bonuses for each rook when there is
  // no (friendly) pawn on the rook file.
  constexpr Score RookOnFile[] = { S(19, 7), S(48, 29) };

  // ThreatByMinor/ByRook[attacked PieceType] contains bonuses according to
  // which piece type attacks which one. Attacks on lesser pieces which are
  // pawn-defended are not considered.
  constexpr Score ThreatByMinor[PIECE_TYPE_NB] = {
    S(0, 0), S(5, 32), S(57, 41), S(77, 56), S(88, 119), S(79, 161)
  };

  constexpr Score ThreatByRook[PIECE_TYPE_NB] = {
    S(0, 0), S(3, 46), S(37, 68), S(42, 60), S(0, 38), S(58, 41)
  };

  // Assorted bonuses and penalties
  constexpr Score BadOutpost          = S( -7, 36);
  constexpr Score BishopOnKingRing    = S( 24,  0);
  constexpr Score BishopPawns         = S(  3,  7);
  constexpr Score BishopXRayPawns     = S(  4,  5);
  constexpr Score CorneredBishop      = S( 50, 50);
  constexpr Score FlankAttacks        = S(  8,  0);
  constexpr Score Hanging             = S( 69, 36);
  constexpr Score KnightOnQueen       = S( 16, 11);
  constexpr Score LongDiagonalBishop  = S( 45,  0);
  constexpr Score MinorBehindPawn     = S( 18,  3);
  constexpr Score PassedFile          = S( 11,  8);
  constexpr Score PawnlessFlank       = S( 17, 95);
  constexpr Score QueenInfiltration   = S( -2, 14);
  constexpr Score ReachableOutpost    = S( 31, 22);
  constexpr Score RestrictedPiece     = S(  7,  7);
  constexpr Score RookOnKingRing      = S( 16,  0);
  constexpr Score RookOnQueenFile     = S(  6, 11);
  constexpr Score SliderOnQueen       = S( 60, 18);
  constexpr Score ThreatByKing        = S( 24, 89);
  constexpr Score ThreatByPawnPush    = S( 48, 39);
  constexpr Score ThreatBySafePawn    = S(173, 94);
  constexpr Score TrappedRook         = S( 55, 13);
  constexpr Score WeakQueenProtection = S( 14,  0);
  constexpr Score WeakQueen           = S( 56, 15);


#undef S

  // Evaluation class computes and stores attacks tables and other working data
  template<Tracing T>
  class Evaluation {

  public:
    Evaluation() = delete;
    explicit Evaluation(const Position& p) : pos(p) {}
    Evaluation& operator=(const Evaluation&) = delete;
    Value value();

  private:
    template<Color Us> void initialize();
    template<Color Us, PieceType Pt> Score pieces();
    template<Color Us>
    [[nodiscard]] Score king() const;
    template<Color Us>
    [[nodiscard]] Score threats() const;
    template<Color Us>
    [[nodiscard]] Score passed() const;
    template<Color Us>
    [[nodiscard]] Score space() const;
    [[nodiscard]] Value winnable(Score score) const;

    const Position& pos;
    Material::Entry* me;
    Pawns::Entry* pe;
    Bitboard mobilityArea[COLOR_NB];
    Score mobility[COLOR_NB] = { SCORE_ZERO, SCORE_ZERO };

    // attackedBy[color][piece type] is a bitboard representing all squares
    // attacked by a given color and piece type. Special "piece types" which
    // is also calculated is ALL_PIECES.
    Bitboard attackedBy[COLOR_NB][PIECE_TYPE_NB];

    // attackedBy2[color] are the squares attacked by at least 2 units of a given
    // color, including x-rays. But diagonal x-rays through pawns are not computed.
    Bitboard attackedBy2[COLOR_NB];

    // kingRing[color] are the squares adjacent to the king plus some other
    // very near squares, depending on king position.
    Bitboard kingRing[COLOR_NB];

    // kingAttackersCount[color] is the number of pieces of the given color
    // which attack a square in the kingRing of the enemy king.
    int kingAttackersCount[COLOR_NB];

    // kingAttackersWeight[color] is the sum of the "weights" of the pieces of
    // the given color which attack a square in the kingRing of the enemy king.
    // The weights of the individual piece types are given by the elements in
    // the KingAttackWeights array.
    int kingAttackersWeight[COLOR_NB];

    // kingAttacksCount[color] is the number of attacks by the given color to
    // squares directly adjacent to the enemy king. Pieces which attack more
    // than one square are counted multiple times. For instance, if there is
    // a white knight on g5 and black's king is on g8, this white knight adds 2
    // to kingAttacksCount[WHITE].
    int kingAttacksCount[COLOR_NB];
  };


  // Evaluation::initialize() computes king and pawn attacks, and the king ring
  // bitboard for a given color. This is done at the beginning of the evaluation.

  template<Tracing T> template<Color Us>
  void Evaluation<T>::initialize() {

    constexpr Color     Them = ~Us;
    constexpr Direction Up   = pawn_push(Us);
    constexpr Direction Down = -Up;
    constexpr Bitboard LowRanks = Us == WHITE ? Rank2BB | Rank3BB : Rank7BB | Rank6BB;

    const Square ksq = pos.square<KING>(Us);

    const Bitboard dblAttackByPawn = pawn_double_attacks_bb<Us>(pos.pieces(Us, PAWN));

    // Find our pawns that are blocked or on the first two ranks
    const Bitboard b = pos.pieces(Us, PAWN) & (shift<Down>(pos.pieces()) | LowRanks);

    // Squares occupied by those pawns, by our king or queen, by blockers to attacks on our king
    // or controlled by enemy pawns are excluded from the mobility area.
    mobilityArea[Us] = ~(b | pos.pieces(Us, KING, QUEEN) | pos.blockers_for_king(Us) | pe->pawn_attacks(Them));

    // Initialize attackedBy[] for king and pawns
    attackedBy[Us][KING] = attacks_bb<KING>(ksq);
    attackedBy[Us][PAWN] = pe->pawn_attacks(Us);
    attackedBy[Us][ALL_PIECES] = attackedBy[Us][KING] | attackedBy[Us][PAWN];
    attackedBy2[Us] = dblAttackByPawn | attackedBy[Us][KING] & attackedBy[Us][PAWN];

    // Init our king safety tables
    const Square s = make_square(Utility::clamp(file_of(ksq), FILE_B, FILE_G),
                                 Utility::clamp(rank_of(ksq), RANK_2, RANK_7));
    kingRing[Us] = attacks_bb<KING>(s) | s;

    kingAttackersCount[Them] = popcount(kingRing[Us] & pe->pawn_attacks(Them));
    kingAttacksCount[Them] = kingAttackersWeight[Them] = 0;

    // Remove from kingRing[] the squares defended by two pawns
    kingRing[Us] &= ~dblAttackByPawn;
  }


  // Evaluation::pieces() scores pieces of a given color and type

  template<Tracing T> template<Color Us, PieceType Pt>
  Score Evaluation<T>::pieces() {

    constexpr Color     Them = ~Us;
    constexpr Direction Down = -pawn_push(Us);
    constexpr Bitboard OutpostRanks = Us == WHITE ? Rank4BB | Rank5BB | Rank6BB
	                                      : Rank5BB | Rank4BB | Rank3BB;
    const Square* pl = pos.squares<Pt>(Us);

    Score score = SCORE_ZERO;

    attackedBy[Us][Pt] = 0;

    for (Square s = *pl; s != SQ_NONE; s = *++pl)
    {
        // Find attacked squares, including x-ray attacks for bishops and rooks
        Bitboard b = Pt == BISHOP
	                     ? attacks_bb<BISHOP>(s, pos.pieces() ^ pos.pieces(QUEEN))
	                     : Pt == ROOK
	                     ? attacks_bb<ROOK>(s, pos.pieces() ^ pos.pieces(QUEEN) ^ pos.pieces(Us, ROOK))
	                     : attacks_bb<Pt>(s, pos.pieces());

        if (pos.blockers_for_king(Us) & s)
            b &= line_bb(pos.square<KING>(Us), s);

        attackedBy2[Us] |= attackedBy[Us][ALL_PIECES] & b;
        attackedBy[Us][Pt] |= b;
        attackedBy[Us][ALL_PIECES] |= b;

        if (b & kingRing[Them])
        {
            ++kingAttackersCount[Us];
            kingAttackersWeight[Us] += KingAttackWeights[Pt];
            kingAttacksCount[Us] += popcount(b & attackedBy[Them][KING]);
        }

        else if (Pt == ROOK && file_bb(s) & kingRing[Them])
            score += RookOnKingRing;

        else if (Pt == BISHOP && attacks_bb<BISHOP>(s, pos.pieces(PAWN)) & kingRing[Them])
            score += BishopOnKingRing;

        const int mob = popcount(b & mobilityArea[Us]);

        mobility[Us] += MobilityBonus[Pt - 2][mob];

        if (Pt == BISHOP || Pt == KNIGHT)
        {
            // Bonus if piece is on an outpost square or can reach one
            if (Bitboard bb = OutpostRanks & attackedBy[Us][PAWN] & ~pe->pawn_attacks_span(Them); Pt == KNIGHT
                && bb & s & ~CenterFiles
                && !(b & pos.pieces(Them) & ~pos.pieces(PAWN))
                && !conditional_more_than_two(
                      pos.pieces(Them) & ~pos.pieces(PAWN) & (s & QueenSide ? QueenSide : KingSide)))
                score += BadOutpost;
            else if (bb & s)
                score += Outpost[Pt == BISHOP];
            else if (Pt == KNIGHT && bb & b & ~pos.pieces(Us))
                score += ReachableOutpost;

            // Bonus for a knight or bishop shielded by pawn
            if (shift<Down>(pos.pieces(PAWN)) & s)
                score += MinorBehindPawn;

            // Penalty if the piece is far from the king
            score -= KingProtector[Pt == BISHOP] * distance(pos.square<KING>(Us), s);

            if (Pt == BISHOP)
            {
                // Penalty according to the number of our pawns on the same color square as the
                // bishop, bigger when the center files are blocked with pawns and smaller
                // when the bishop is outside the pawn chain.
                Bitboard blocked = pos.pieces(Us, PAWN) & shift<Down>(pos.pieces());

                score -= BishopPawns * pos.pawns_on_same_color_squares(Us, s)
                                     * (!(attackedBy[Us][PAWN] & s) + popcount(blocked & CenterFiles));

                // Penalty for all enemy pawns x-rayed
                score -= BishopXRayPawns * popcount(attacks_bb<BISHOP>(s) & pos.pieces(Them, PAWN));

                // Bonus for bishop on a long diagonal which can "see" both center squares
                if (more_than_one(attacks_bb<BISHOP>(s, pos.pieces(PAWN)) & Center))
                    score += LongDiagonalBishop;

                // An important Chess960 pattern: a cornered bishop blocked by a friendly
                // pawn diagonally in front of it is a very serious problem, especially
                // when that pawn is also blocked.
                if (   pos.is_chess960()
                    && (s == relative_square(Us, SQ_A1) || s == relative_square(Us, SQ_H1)))
                {
	                if (const Direction d = pawn_push(Us) + (file_of(s) == FILE_A ? EAST : WEST); pos.piece_on(s + d) == make_piece(Us, PAWN))
                        score -= !pos.empty(s + d + pawn_push(Us))                ? CorneredBishop * 4
                                : pos.piece_on(s + d + d) == make_piece(Us, PAWN) ? CorneredBishop * 2
                                                                                  : CorneredBishop;
                }
            }
        }

        if (Pt == ROOK)
        {
            // Bonus for rook on the same file as a queen
            if (file_bb(s) & pos.pieces(QUEEN))
                score += RookOnQueenFile;

            // Bonus for rook on an open or semi-open file
            if (pos.is_on_semiopen_file(Us, s))
                score += RookOnFile[pos.is_on_semiopen_file(Them, s)];

            // Penalty when trapped by the king, even more if the king cannot castle
            else if (mob <= 3)
            {
	            if (const File kf = file_of(pos.square<KING>(Us)); kf < FILE_E == file_of(s) < kf)
                    score -= TrappedRook * (1 + !pos.castling_rights(Us));
            }
        }

        if (Pt == QUEEN)
        {
            // Penalty if any relative pin or discovered attack against the queen
            if (Bitboard queenPinners; pos.slider_blockers(pos.pieces(Them, ROOK, BISHOP), s, queenPinners))
                score -= WeakQueen;

            // Bonus for queen on weak square in enemy camp
            if (relative_rank(Us, s) > RANK_4 && ~pe->pawn_attacks_span(Them) & s)
                score += QueenInfiltration;
        }
    }
    if (T)
	    add(Pt, Us, score);

    return score;
  }


  // Evaluation::king() assigns bonuses and penalties to a king of a given color

  template<Tracing T> template<Color Us>
  [[nodiscard]] [[nodiscard]] Score Evaluation<T>::king() const {

    constexpr Color    Them = ~Us;
    constexpr Bitboard Camp = Us == WHITE ? AllSquares ^ Rank6BB ^ Rank7BB ^ Rank8BB
	                              : AllSquares ^ Rank1BB ^ Rank2BB ^ Rank3BB;

    Bitboard unsafeChecks = 0;
    int kingDanger = 0;
    const Square ksq = pos.square<KING>(Us);

    // Init the score with king shelter and enemy pawns storm
    Score score = pe->king_safety<Us>(pos);

    // Attacked squares defended at most once by our queen or king
    Bitboard weak = attackedBy[Them][ALL_PIECES]
	    & ~attackedBy2[Us]
	    & (~attackedBy[Us][ALL_PIECES] | attackedBy[Us][KING] | attackedBy[Us][QUEEN]);

    // Analyse the safe enemy's checks which are possible on next move
    Bitboard safe = ~pos.pieces(Them);
    safe &= ~attackedBy[Us][ALL_PIECES] | weak & attackedBy2[Them];

    Bitboard b1 = attacks_bb<ROOK>(ksq, pos.pieces() ^ pos.pieces(Us, QUEEN));
    Bitboard b2 = attacks_bb<BISHOP>(ksq, pos.pieces() ^ pos.pieces(Us, QUEEN));

    // Enemy rooks checks
    const Bitboard rookChecks = b1 & attackedBy[Them][ROOK] & safe;
    if (rookChecks)
        kingDanger += SafeCheck[ROOK][more_than_one(rookChecks)];
    else
        unsafeChecks |= b1 & attackedBy[Them][ROOK];

    // Enemy queen safe checks: count them only if the checks are from squares from
    // which opponent cannot give a rook check, because rook checks are more valuable.
    const Bitboard queenChecks = (b1 | b2) & attackedBy[Them][QUEEN] & safe
	    & ~(attackedBy[Us][QUEEN] | rookChecks);
    if (queenChecks)
        kingDanger += SafeCheck[QUEEN][more_than_one(queenChecks)];

    // Enemy bishops checks: count them only if they are from squares from which
    // opponent cannot give a queen check, because queen checks are more valuable.
    if (const Bitboard bishopChecks = b2 & attackedBy[Them][BISHOP] & safe
	    & ~queenChecks)
        kingDanger += SafeCheck[BISHOP][more_than_one(bishopChecks)];

    else
        unsafeChecks |= b2 & attackedBy[Them][BISHOP];

    // Enemy knights checks
    if (Bitboard knightChecks = attacks_bb<KNIGHT>(ksq) & attackedBy[Them][KNIGHT]; knightChecks & safe)
        kingDanger += SafeCheck[KNIGHT][more_than_one(knightChecks & safe)];
    else
        unsafeChecks |= knightChecks;

    // Find the squares that opponent attacks in our king flank, the squares
    // which they attack twice in that flank, and the squares that we defend.
    b1 = attackedBy[Them][ALL_PIECES] & KingFlank[file_of(ksq)] & Camp;
    b2 = b1 & attackedBy2[Them];
    const Bitboard b3 = attackedBy[Us][ALL_PIECES] & KingFlank[file_of(ksq)] & Camp;

    const int kingFlankAttack  = popcount(b1) + popcount(b2);
    const int kingFlankDefense = popcount(b3);

    kingDanger +=        kingAttackersCount[Them] * kingAttackersWeight[Them]
                 + 185 * popcount(kingRing[Us] & weak)
                 + 148 * popcount(unsafeChecks)
                 +  98 * popcount(pos.blockers_for_king(Us))
                 +  69 * kingAttacksCount[Them]
                 +   3 * kingFlankAttack * kingFlankAttack / 8
                 +       mg_value(mobility[Them] - mobility[Us])
                 - 873 * !pos.count<QUEEN>(Them)
                 - 100 * static_cast<bool>(attackedBy[Us][KNIGHT] & attackedBy[Us][KING])
                 -   6 * mg_value(score) / 8
                 -   4 * kingFlankDefense
                 +  37;

    // Transform the kingDanger units into a Score, and subtract it from the evaluation
    if (kingDanger > 100)
        score -= make_score(kingDanger * kingDanger / 4096, kingDanger / 16);

    // Penalty when our king is on a pawnless flank
    if (!(pos.pieces(PAWN) & KingFlank[file_of(ksq)]))
        score -= PawnlessFlank;

    // Penalty if king flank is under attack, potentially moving toward the king
    score -= FlankAttacks * kingFlankAttack;

    if (T)
	    add(KING, Us, score);

    return score;
  }


  // Evaluation::threats() assigns bonuses according to the types of the
  // attacking and the attacked pieces.

  template<Tracing T> template<Color Us>
  [[nodiscard]] [[nodiscard]] Score Evaluation<T>::threats() const {

    constexpr Color     Them     = ~Us;
    constexpr Direction Up       = pawn_push(Us);
    constexpr Bitboard  TRank3BB = Us == WHITE ? Rank3BB : Rank6BB;

    Bitboard b;
    Score score = SCORE_ZERO;

    // Non-pawn enemies
    Bitboard nonPawnEnemies = pos.pieces(Them) & ~pos.pieces(PAWN);

    // Squares strongly protected by the enemy, either because they defend the
    // square with a pawn, or because they defend the square twice and we don't.
    const Bitboard stronglyProtected = attackedBy[Them][PAWN]
	    | attackedBy2[Them] & ~attackedBy2[Us];

    // Non-pawn enemies, strongly protected
    const Bitboard defended = nonPawnEnemies & stronglyProtected;

    // Enemies not strongly protected and under our attack

    // Bonus according to the kind of attacking pieces
    if (Bitboard weak = pos.pieces(Them) & ~stronglyProtected & attackedBy[Us][ALL_PIECES]; defended | weak)
    {
        b = (defended | weak) & (attackedBy[Us][KNIGHT] | attackedBy[Us][BISHOP]);
        while (b)
            score += ThreatByMinor[type_of(pos.piece_on(pop_lsb(&b)))];

        b = weak & attackedBy[Us][ROOK];
        while (b)
            score += ThreatByRook[type_of(pos.piece_on(pop_lsb(&b)))];

        if (weak & attackedBy[Us][KING])
            score += ThreatByKing;

        b =  ~attackedBy[Them][ALL_PIECES]
           | nonPawnEnemies & attackedBy2[Us];
        score += Hanging * popcount(weak & b);

        // Additional bonus if weak piece is only protected by a queen
        score += WeakQueenProtection * popcount(weak & attackedBy[Them][QUEEN]);
    }

    // Bonus for restricting their piece moves
    b =   attackedBy[Them][ALL_PIECES]
       & ~stronglyProtected
       &  attackedBy[Us][ALL_PIECES];
    score += RestrictedPiece * popcount(b);

    // Protected or unattacked squares
    Bitboard safe = ~attackedBy[Them][ALL_PIECES] | attackedBy[Us][ALL_PIECES];

    // Bonus for attacking enemy pieces with our relatively safe pawns
    b = pos.pieces(Us, PAWN) & safe;
    b = pawn_attacks_bb<Us>(b) & nonPawnEnemies;
    score += ThreatBySafePawn * popcount(b);

    // Find squares where our pawns can push on the next move
    b  = shift<Up>(pos.pieces(Us, PAWN)) & ~pos.pieces();
    b |= shift<Up>(b & TRank3BB) & ~pos.pieces();

    // Keep only the squares which are relatively safe
    b &= ~attackedBy[Them][PAWN] & safe;

    // Bonus for safe pawn threats on the next move
    b = pawn_attacks_bb<Us>(b) & nonPawnEnemies;
    score += ThreatByPawnPush * popcount(b);

    // Bonus for threats on the next moves against enemy queen
    if (pos.count<QUEEN>(Them) == 1)
    {
	    const Square s = pos.square<QUEEN>(Them);
        safe = mobilityArea[Us] & ~stronglyProtected;

        b = attackedBy[Us][KNIGHT] & attacks_bb<KNIGHT>(s);

        score += KnightOnQueen * popcount(b & safe);

        b =  attackedBy[Us][BISHOP] & attacks_bb<BISHOP>(s, pos.pieces())
           | attackedBy[Us][ROOK  ] & attacks_bb<ROOK  >(s, pos.pieces());

        score += SliderOnQueen * popcount(b & safe & attackedBy2[Us]);
    }

    if (T)
	    add(THREAT, Us, score);

    return score;
  }

  // Evaluation::passed() evaluates the passed pawns and candidate passed
  // pawns of the given color.

  template<Tracing T> template<Color Us>
  [[nodiscard]] [[nodiscard]] Score Evaluation<T>::passed() const {

    constexpr Color     Them = ~Us;
    constexpr Direction Up   = pawn_push(Us);
    constexpr Direction Down = -Up;

    auto king_proximity = [&](const Color c, const Square s) {
      return std::min(distance(pos.square<KING>(c), s), 5);
    };

    Bitboard b;
    Score score = SCORE_ZERO;

    b = pe->passed_pawns(Us);

    if (const Bitboard blockedPassers = b & shift<Down>(pos.pieces(Them, PAWN)))
    {
	    const Bitboard helpers = shift<Up>(pos.pieces(Us, PAWN))
	        & ~pos.pieces(Them)
	        & (~attackedBy2[Them] | attackedBy[Us][ALL_PIECES]);

        // Remove blocked candidate passers that don't have help to pass
        b &=  ~blockedPassers
            | shift<WEST>(helpers)
            | shift<EAST>(helpers);
    }

    while (b)
    {
	    const Square s = pop_lsb(&b);

        assert(!(pos.pieces(Them, PAWN) & forward_file_bb(Us, s + Up)));

	    const int r = relative_rank(Us, s);

        Score bonus = PassedRank[r];

        if (r > RANK_3)
        {
            int w = 5 * r - 13;
            Square blockSq = s + Up;

            // Adjust bonus based on the king's proximity
            bonus += make_score(0, (  king_proximity(Them, blockSq) * 19 / 4
                                     - king_proximity(Us,   blockSq) *  2) * w);

            // If blockSq is not the queening square then consider also a second push
            if (r != RANK_7)
                bonus -= make_score(0, king_proximity(Us, blockSq + Up) * w);

            // If the pawn is free to advance, then increase the bonus
            if (pos.empty(blockSq))
            {
	            const Bitboard squaresToQueen = forward_file_bb(Us, s);
                Bitboard unsafeSquares = passed_pawn_span(Us, s);

	            const Bitboard bb = forward_file_bb(Them, s) & pos.pieces(ROOK, QUEEN);

                if (!(pos.pieces(Them) & bb))
                    unsafeSquares &= attackedBy[Them][ALL_PIECES];

                // If there are no enemy attacks on passed pawn span, assign a big bonus.
                // Otherwise assign a smaller bonus if the path to queen is not attacked
                // and even smaller bonus if it is attacked but block square is not.
                int k = !unsafeSquares                    ? 35 :
                        !(unsafeSquares & squaresToQueen) ? 20 :
                        !(unsafeSquares & blockSq)        ?  9 :
                                                             0 ;

                // Assign a larger bonus if the block square is defended
                if (pos.pieces(Us) & bb || attackedBy[Us][ALL_PIECES] & blockSq)
                    k += 5;

                bonus += make_score(k * w, k * w);
            }
        } // r > RANK_3

        score += bonus - PassedFile * edge_distance(file_of(s));
    }

    if (T)
	    add(PASSED, Us, score);

    return score;
  }


  // Evaluation::space() computes a space evaluation for a given side, aiming to improve game
  // play in the opening. It is based on the number of safe squares on the 4 central files
  // on ranks 2 to 4. Completely safe squares behind a friendly pawn are counted twice.
  // Finally, the space bonus is multiplied by a weight which decreases according to occupancy.

  template<Tracing T> template<Color Us>
  [[nodiscard]] [[nodiscard]] Score Evaluation<T>::space() const {

    // Early exit if, for example, both queens or 6 minor pieces have been exchanged
    if (pos.non_pawn_material() < SpaceThreshold)
        return SCORE_ZERO;

    constexpr Color Them     = ~Us;
    constexpr Direction Down = -pawn_push(Us);
    constexpr Bitboard SpaceMask =
      Us == WHITE ? CenterFiles & (Rank2BB | Rank3BB | Rank4BB)
                  : CenterFiles & (Rank7BB | Rank6BB | Rank5BB);

    // Find the available squares for our pieces inside the area defined by SpaceMask
    const Bitboard safe =   SpaceMask
                   & ~pos.pieces(Us, PAWN)
                   & ~attackedBy[Them][PAWN];

    // Find all squares which are at most three squares behind some friendly pawn
    Bitboard behind = pos.pieces(Us, PAWN);
    behind |= shift<Down>(behind);
    behind |= shift<Down+Down>(behind);

    const int bonus = popcount(safe) + popcount(behind & safe & ~attackedBy[Them][ALL_PIECES]);
    const int weight = pos.count<ALL_PIECES>(Us) - 3 + std::min(pe->blocked_count(), 9);
    const Score score = make_score(bonus * weight * weight / 16, 0);

    if (T)
	    add(SPACE, Us, score);

    return score;
  }


  // Evaluation::winnable() adjusts the midgame and endgame score components, based on
  // the known attacking/defending status of the players. The final value is derived
  // by interpolation from the midgame and endgame values.

  template<Tracing T>
  Value Evaluation<T>::winnable(const Score score) const {
	  const int outflanking =  distance<File>(pos.square<KING>(WHITE), pos.square<KING>(BLACK))
                     - distance<Rank>(pos.square<KING>(WHITE), pos.square<KING>(BLACK));

	  const bool pawnsOnBothFlanks =   pos.pieces(PAWN) & QueenSide
                            && pos.pieces(PAWN) & KingSide;

	  const bool almostUnwinnable =   outflanking < 0
                           && !pawnsOnBothFlanks;

	  const bool infiltration =   rank_of(pos.square<KING>(WHITE)) > RANK_4
                       || rank_of(pos.square<KING>(BLACK)) < RANK_5;

    // Compute the initiative bonus for the attacking side
	  const int complexity =   9 * pe->passed_count()
                    + 12 * pos.count<PAWN>()
                    +  9 * outflanking
                    + 21 * pawnsOnBothFlanks
                    + 24 * infiltration
                    + 51 * !pos.non_pawn_material()
                    - 43 * almostUnwinnable
                    -110 ;

    Value mg = mg_value(score);
    Value eg = eg_value(score);

    // Now apply the bonus: note that we find the attacking side by extracting the
    // sign of the midgame or endgame values, and that we carefully cap the bonus
    // so that the midgame and endgame scores do not change sign after the bonus.
	  const int u = ((mg > 0) - (mg < 0)) * Utility::clamp(complexity + 50, -abs(mg), 0);
    int v = ((eg > 0) - (eg < 0)) * std::max(complexity, -abs(eg));

    mg += u;
    eg += v;

    // Compute the scale factor for the winning side
	  const Color strongSide = eg > VALUE_DRAW ? WHITE : BLACK;
    int sf = me->scale_factor(pos, strongSide);

    // If scale factor is not already specific, scale down via general heuristics
    if (sf == SCALE_FACTOR_NORMAL)
    {
        if (pos.opposite_bishops())
        {
            if (   pos.non_pawn_material(WHITE) == BishopValueMg
                && pos.non_pawn_material(BLACK) == BishopValueMg)
                sf = 18 + 4 * popcount(pe->passed_pawns(strongSide));
            else
                sf = 22 + 3 * pos.count<ALL_PIECES>(strongSide);
        }
        else if (  pos.non_pawn_material(WHITE) == RookValueMg
                && pos.non_pawn_material(BLACK) == RookValueMg
                && pos.count<PAWN>(strongSide) - pos.count<PAWN>(~strongSide) <= 1
                && static_cast<bool>(KingSide & pos.pieces(strongSide, PAWN)) != static_cast<bool>(QueenSide & pos.pieces(strongSide, PAWN))
                && attackedBy[~strongSide][KING] & pos.pieces(~strongSide, PAWN))
            sf = 36;
        else if (pos.count<QUEEN>() == 1)
            sf = 37 + 3 * (pos.count<QUEEN>(WHITE) == 1 ? pos.count<BISHOP>(BLACK) + pos.count<KNIGHT>(BLACK)
                                                        : pos.count<BISHOP>(WHITE) + pos.count<KNIGHT>(WHITE));
        else
            sf = std::min(sf, 36 + 7 * pos.count<PAWN>(strongSide));
    }

    // Interpolate between the middlegame and (scaled by 'sf') endgame score
    v =  mg * me->game_phase()
       + eg * (PHASE_MIDGAME - me->game_phase()) * sf / SCALE_FACTOR_NORMAL;
    v /= PHASE_MIDGAME;

    if (T)
    {
	    add(WINNABLE, make_score(u, eg * sf / SCALE_FACTOR_NORMAL - eg_value(score)));
	    add(TOTAL, make_score(mg, eg * sf / SCALE_FACTOR_NORMAL));
    }

    return static_cast<Value>(v);
  }


  // Evaluation::value() is the main function of the class. It computes the various
  // parts of the evaluation and returns the value of the position from the point
  // of view of the side to move.

  template<Tracing T>
  Value Evaluation<T>::value() {

    assert(!pos.checkers());

    // Probe the material hash table
    me = Material::probe(pos);

    // If we have a specialized evaluation function for the current material
    // configuration, call it and return.
    if (me->specialized_eval_exists())
        return me->evaluate(pos);

    // Initialize score by reading the incrementally updated scores included in
    // the position object (material + piece square tables) and the material
    // imbalance. Score is computed internally from the white point of view.
    Score score = pos.psq_score() + me->imbalance() + pos.this_thread()->contempt;

    // Probe the pawn hash table
    pe = Pawns::probe(pos);
    score += pe->pawn_score(WHITE) - pe->pawn_score(BLACK);

    // Early exit if score is high
    Value v = (mg_value(score) + eg_value(score)) / 2;
    if (abs(v) > LazyThreshold + pos.non_pawn_material() / 64)
       return pos.side_to_move() == WHITE ? v : -v;

    // Main evaluation begins here
    initialize<WHITE>();
    initialize<BLACK>();

    // Pieces evaluated first (also populates attackedBy, attackedBy2).
    // Note that the order of evaluation of the terms is left unspecified.
    score +=  pieces<WHITE, KNIGHT>() - pieces<BLACK, KNIGHT>()
            + pieces<WHITE, BISHOP>() - pieces<BLACK, BISHOP>()
            + pieces<WHITE, ROOK  >() - pieces<BLACK, ROOK  >()
            + pieces<WHITE, QUEEN >() - pieces<BLACK, QUEEN >();

    score += mobility[WHITE] - mobility[BLACK];

    // More complex interactions that require fully populated attack bitboards
    score +=  king<   WHITE>() - king<   BLACK>()
            + threats<WHITE>() - threats<BLACK>()
            + passed< WHITE>() - passed< BLACK>()
            + space<  WHITE>() - space<  BLACK>();

    // Derive single value from mg and eg parts of score
    v = winnable(score);

    // In case of tracing add all remaining individual evaluation terms
    if (T)
    {
	    add(MATERIAL, pos.psq_score());
	    add(IMBALANCE, me->imbalance());
	    add(PAWN, pe->pawn_score(WHITE), pe->pawn_score(BLACK));
	    add(MOBILITY, mobility[WHITE], mobility[BLACK]);
    }

    // Evaluation grain
    v = v / 16 * 16;

    // Side to move point of view
    v = (pos.side_to_move() == WHITE ? v : -v) + Tempo;

    // Damp down the evaluation linearly when shuffling
    v = v * (100 - pos.rule50_count()) / 100;

    return v;
  }

} // namespace


/// evaluate() is the evaluator for the outer world. It returns a static
/// evaluation of the position from the point of view of the side to move.

Value Eval::evaluate(const Position& pos) {
  if (static_cast<size_t>(Options["EvalNNUE"]))
  	return NNUE::evaluate(pos);
  return Evaluation<NO_TRACE>(pos).value();
}


/// trace() is like evaluate(), but instead of returning a value, it returns
/// a string (suitable for outputting to stdout) that contains the detailed
/// descriptions and values of each evaluation term. Useful for debugging.

std::string Eval::trace(const Position& pos) {

  if (pos.checkers())
      return "Total evaluation: none (in check)";

  std::memset(scores, 0, sizeof scores);

  pos.this_thread()->contempt = SCORE_ZERO; // Reset any dynamic contempt

  Value v = Evaluation<TRACE>(pos).value();

  v = pos.side_to_move() == WHITE ? v : -v; // Trace scores are from white's point of view

  std::stringstream ss;
  ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2)
     << "     Term    |    White    |    Black    |    Total   \n"
     << "             |   MG    EG  |   MG    EG  |   MG    EG \n"
     << " ------------+-------------+-------------+------------\n"
     << "    Material | " << MATERIAL
     << "   Imbalance | " << IMBALANCE
     << "       Pawns | " << static_cast<Term>(PAWN)
     << "     Knights | " << static_cast<Term>(KNIGHT)
     << "     Bishops | " << static_cast<Term>(BISHOP)
     << "       Rooks | " << static_cast<Term>(ROOK)
     << "      Queens | " << static_cast<Term>(QUEEN)
     << "    Mobility | " << MOBILITY
     << " King safety | " << static_cast<Term>(KING)
     << "     Threats | " << THREAT
     << "      Passed | " << PASSED
     << "       Space | " << SPACE
     << "    Winnable | " << WINNABLE
     << " ------------+-------------+-------------+------------\n"
     << "       Total | " << TOTAL;

  ss << "\nFinal evaluation: " << to_cp(v) << " (white side)\n";

  return ss.str();
}

#if defined(EVAL_NNUE) || defined(EVAL_LEARN)
namespace Eval {
ExtBonaPiece kpp_board_index[PIECE_NB] = {
    { BONA_PIECE_ZERO, BONA_PIECE_ZERO },
    { f_pawn, e_pawn },
    { f_knight, e_knight },
    { f_bishop, e_bishop },
    { f_rook, e_rook },
    { f_queen, e_queen },
    { f_king, e_king },
    { BONA_PIECE_ZERO, BONA_PIECE_ZERO },

    // When viewed from behind. f and e are exchanged.
    { BONA_PIECE_ZERO, BONA_PIECE_ZERO },
    { e_pawn, f_pawn },
    { e_knight, f_knight },
    { e_bishop, f_bishop },
    { e_rook, f_rook },
    { e_queen, f_queen },
    { e_king, f_king },
    { BONA_PIECE_ZERO, BONA_PIECE_ZERO }, // no money
};

// Check whether the pieceListFw[] held internally is a correct BonaPiece.
// Note: For debugging. slow.
bool EvalList::is_valid(const Position& pos) const
{
  std::set<PieceNumber> piece_numbers;
  for (Square sq = SQ_A1; sq != SQUARE_NB; ++sq) {
    auto piece_number = piece_no_of_board(sq);
    if (piece_number == PIECE_NUMBER_NB) {
      continue;
    }
    assert(!piece_numbers.count(piece_number));
    piece_numbers.insert(piece_number);
  }

  for (int i = 0; i < length(); ++i)
  {
	  const BonaPiece fw = pieceListFw[i];
    // Go to the Position class to see if this fw really exists.

    if (fw == BONA_PIECE_ZERO) {
      continue;
    }

    // Out of range
    if (!(0 <= fw && fw < fe_end))
      return false;

    // Since it is a piece on the board, I will check if this piece really exists.
    for (Piece pc = NO_PIECE; pc < PIECE_NB; ++pc)
    {
	    if (const auto pt = type_of(pc); pt == NO_PIECE_TYPE || pt == 7) // non-existing piece
        continue;

      // BonaPiece start number of piece pc
	    if (const auto s = kpp_board_index[pc].fw; s <= fw && fw < s + SQUARE_NB)
      {
        // Since it was found, check if this piece is at sq.
        const auto sq = static_cast<Square>(fw - s);

        if (const Piece pc2 = pos.piece_on(sq); pc2 != pc)
          return false;

        goto Found;
      }
    }
    // It was a piece that did not exist for some reason..
    return false;
  Found:;
  }

  // Validate piece_no_list_board
  for (auto sq = SQUARE_ZERO; sq < SQUARE_NB; ++sq) {
	  const Piece expected_piece = pos.piece_on(sq);
	  const PieceNumber piece_number = piece_no_list_board[sq];
    if (piece_number == PIECE_NUMBER_NB) {
      assert(expected_piece == NO_PIECE);
      if (expected_piece != NO_PIECE) {
        return false;
      }
      continue;
    }

	  const BonaPiece bona_piece_white = pieceListFw[piece_number];
    Piece actual_piece;
    for (actual_piece = NO_PIECE; actual_piece < PIECE_NB; ++actual_piece) {
      if (kpp_board_index[actual_piece].fw == BONA_PIECE_ZERO) {
        continue;
      }

      if (kpp_board_index[actual_piece].fw <= bona_piece_white
        && bona_piece_white < kpp_board_index[actual_piece].fw + SQUARE_NB) {
        break;
      }
    }

    assert(actual_piece != PIECE_NB);
    if (actual_piece == PIECE_NB) {
      return false;
    }

    assert(actual_piece == expected_piece);
    if (actual_piece != expected_piece) {
      return false;
    }

	  const auto actual_square = static_cast<Square>(
      bona_piece_white - kpp_board_index[actual_piece].fw);
    assert(sq == actual_square);
    if (sq != actual_square) {
      return false;
    }
  }

  return true;
}
}
#endif  // defined(EVAL_NNUE) || defined(EVAL_LEARN)

#if !defined(EVAL_NNUE)
namespace Eval {
void evaluate_with_no_return(const Position& pos) {}
void update_weights(uint64_t epoch, const std::array<bool, 4> & freeze) {}
void init_grad(double eta1, uint64_t eta_epoch, double eta2, uint64_t eta2_epoch, double eta3) {}
void add_grad(Position& pos, Color rootColor, double delt_grad, const std::array<bool, 4> & freeze) {}
void save_eval(std::string suffix) {}
double get_eta() { return 0.0; }
}
#endif  // defined(EVAL_NNUE)
