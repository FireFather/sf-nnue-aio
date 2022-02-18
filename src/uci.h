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

#ifndef UCI_H_INCLUDED
#define UCI_H_INCLUDED

#include <map>
#include <string>

class Position;

namespace UCI {

class Option;

/// Custom comparator because UCI options should be case insensitive
struct CaseInsensitiveLess {
  bool operator() (const std::string&, const std::string&) const;
};

/// Our options container is actually a std::map
typedef std::map<std::string, Option, CaseInsensitiveLess> OptionsMap;

/// Option class implements an option as defined by UCI protocol
class Option {

  typedef void (*OnChange)(const Option&);

public:
  explicit Option(OnChange = nullptr);
  explicit Option(bool v, OnChange = nullptr);
  explicit Option(const char* v, OnChange = nullptr);
  Option(double v, int minv, int maxv, OnChange = nullptr);
  Option(const char* v, const char* cur, OnChange = nullptr);

  Option& operator=(const std::string&);
  void operator<<(const Option&);
  operator double() const;
  operator std::string() const;
  bool operator==(const char*) const;

private:
  friend std::ostream& operator<<(std::ostream&, const OptionsMap&);

  std::string defaultValue, currentValue, type;
  int min, max;
  size_t idx{};
  OnChange on_change;
};

void init(OptionsMap&);
void loop(int argc, char* argv[]);
std::string value(Value v);
std::string square(Square s);
std::string move(Move m, bool chess960);
std::string pv(const Position& pos, Depth depth, Value alpha, Value beta);
std::string wdl(Value v, int ply);
Move to_move(const Position& pos, std::string& str);

// Flag that read the evaluation function. This is set to false when evaldir is changed.
extern bool load_eval_finished; // = false;
} // namespace UCI

extern UCI::OptionsMap Options;

// Processing when USI "isready" command is called. At this time, the evaluation function is read.
// Used when you want to load the evaluation function when "isready" does not come in handler of benchmark command etc.
// If skipCorruptCheck == true, skip memory corruption check by check sum when reading the evaluation function a second time.
// * This function is inconvenient if it is not available in Stockfish, so add it.

void init_nnue(bool skipCorruptCheck = false);

extern const char* StartFEN;

#endif // #ifndef UCI_H_INCLUDED
