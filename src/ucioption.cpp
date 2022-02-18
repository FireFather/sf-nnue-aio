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
#include <ostream>
#include <sstream>

#include "misc.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"
#include "syzygy/tbprobe.h"

using std::string;

UCI::OptionsMap Options; // Global object

namespace UCI {

/// 'On change' actions, triggered by an option's value change
void on_clear_hash(const Option&) { Search::clear(); }
void on_hash_size(const Option& o) { TT.resize(static_cast<size_t>(o)); }
void on_logger(const Option& o) { start_logger(o); }
void on_threads(const Option& o) { Threads.set(static_cast<size_t>(o)); }
void on_tb_path(const Option& o) { Tablebases::init(o); }
void on_eval_file(const Option& o)
{
    if (static_cast<bool>(Options["EvalNNUE"]))
    {
    load_eval_finished = false;
    init_nnue();
    }
}


/// Our case insensitive less() function as required by UCI protocol
bool CaseInsensitiveLess::operator() (const string& s1, const string& s2) const {

  return std::lexicographical_compare(s1.begin(), s1.end(), s2.begin(), s2.end(),
         [](const char c1, const char c2) { return tolower(c1) < tolower(c2); });
}


/// UCI::init() initializes the UCI options to their hard-coded default values

void init(OptionsMap& o) {

  constexpr int MaxHashMB = Is64Bit ? 33554432 : 2048;

  o["Debug Log File"]        << Option("", on_logger);
  o["Contempt"]              << Option(24, -100, 100);
  o["Analysis Contempt"]     << Option("Both var Off var White var Black var Both", "Both");
  o["Threads"]               << Option(1, 1, 512, on_threads);
  o["Hash"]                  << Option(16, 1, MaxHashMB, on_hash_size);
  o["Clear Hash"]            << Option(on_clear_hash);
  o["MultiPV"]               << Option(1, 1, 500);
  o["Skill Level"]           << Option(20, 0, 20);
  o["Move Overhead"]         << Option(10, 0, 5000);
  o["Slow Mover"]            << Option(100, 10, 1000);
  o["nodestime"]             << Option(0, 0, 10000);
  // how many moves to use a fixed move
  o["BookMoves"]             << Option(16, 0, 10000);
  o["Ponder"]                << Option(false);
  o["UCI_Chess960"]          << Option(false);
  o["UCI_AnalyseMode"]       << Option(false);
  o["UCI_LimitStrength"]     << Option(false);
  o["UCI_Elo"]               << Option(1350, 1350, 2850);
  o["UCI_ShowWDL"]           << Option(false);
  o["Syzygy50MoveRule"]      << Option(true);
  o["SyzygyPath"]            << Option("<empty>", on_tb_path);
  o["SyzygyProbeDepth"]      << Option(1, 1, 100);
  o["SyzygyProbeLimit"]      << Option(7, 0, 7);
  // Evaluation function file name. When this is changed, it is necessary to reread the evaluation function at the next ucinewgame timing.
  // Without the preceding "./", some GUIs can not load he net file.
  o["EvalFile"]              << Option("./eval/nn.bin", on_eval_file);
#if defined(EVAL_LEARN)
  // When learning the evaluation function, you can change the folder to save the evaluation function.
  // Evalsave by default. This folder shall be prepared in advance.
  // Automatically dig a folder under this folder like "0/", "1/", ... and save the evaluation function file there.
  o["EvalSaveDir"] << Option("evalsave");
#endif
  // When the evaluation function is loaded at the ucinewgame timing, it is necessary to convert the new evaluation function.
  // I want to hit the test eval convert command, but there is no new evaluation function
  // It ends abnormally before executing this command.
  // Therefore, with this hidden option, you can suppress the loading of the evaluation function when ucinewgame,
  // Hit the test eval convert command.
  o["SkipLoadingEval"]       << Option(false);
  o["EvalNNUE"]              << Option(true);
  o["UseEvalHash"]           << Option(false);  
}


/// operator<<() is used to print all the options default values in chronological
/// insertion order (the idx field) and in the format defined by the UCI protocol.

std::ostream& operator<<(std::ostream& os, const OptionsMap& om) {

  for (size_t idx = 0; idx < om.size(); ++idx)
      for (const auto& [fst, snd] : om)
          if (snd.idx == idx)
          {
              const Option& o = snd;
              os << "\noption name " << fst << " type " << o.type;

              if (o.type == "string" || o.type == "check" || o.type == "combo")
                  os << " default " << o.defaultValue;

              if (o.type == "spin")
                  os << " default " << static_cast<int>(stof(o.defaultValue))
                     << " min "     << o.min
                     << " max "     << o.max;

              break;
          }

  return os;
}


/// Option class constructors and conversion operators

Option::Option(const char* v, const OnChange f) : type("string"), min(0), max(0), on_change(f)
{ defaultValue = currentValue = v; }

Option::Option(const bool v, const OnChange f) : type("check"), min(0), max(0), on_change(f)
{ defaultValue = currentValue = v ? "true" : "false"; }

Option::Option(const OnChange f) : type("button"), min(0), max(0), on_change(f)
{}

Option::Option(const double v, const int minv, const int maxv, const OnChange f) : type("spin"), min(minv), max(maxv), on_change(f)
{ defaultValue = currentValue = std::to_string(v); }

Option::Option(const char* v, const char* cur, const OnChange f) : defaultValue(v), currentValue(cur), type("combo"), min(0), max(0), on_change(f)
{
}

Option::operator double() const {
  assert(type == "check" || type == "spin");
  return type == "spin" ? stof(currentValue) : currentValue == "true";
}

Option::operator std::string() const {
  assert(type == "string");
  return currentValue;
}

bool Option::operator==(const char* s) const {
  assert(type == "combo");
  return   !CaseInsensitiveLess()(currentValue, s)
        && !CaseInsensitiveLess()(s, currentValue);
}


/// operator<<() inits options and assigns idx in the correct printing order

void Option::operator<<(const Option& o) {

  static size_t insert_order = 0;

  *this = o;
  idx = insert_order++;
}


/// operator=() updates currentValue and triggers on_change() action. It's up to
/// the GUI to check for option's limits, but we could receive the new value
/// from the user by console window, so let's check the bounds anyway.

Option& Option::operator=(const string& v) {

  assert(!type.empty());

  if (   type != "button" && v.empty()
      || type == "check" && v != "true" && v != "false"
      || type == "spin" && (stof(v) < min || stof(v) > max))
      return *this;

  if (type == "combo")
  {
      OptionsMap comboMap; // To have case insensitive compare
      string token;
      std::istringstream ss(defaultValue);
      while (ss >> token)
          comboMap[token] << Option();
      if (!comboMap.count(v) || v == "var")
          return *this;
  }

  if (type != "button")
      currentValue = v;

  if (on_change)
      on_change(*this);

  return *this;
}

// Flag that read the evaluation function. This is set to false when evaldir is changed.
bool load_eval_finished = false;
} // namespace UCI
