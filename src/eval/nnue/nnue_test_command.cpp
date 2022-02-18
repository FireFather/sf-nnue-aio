﻿// USI extended command for NNUE evaluation function

#if defined(ENABLE_TEST_CMD) && defined(EVAL_NNUE)

#include "../../thread.h"
#include "../../uci.h"
#include "evaluate_nnue.h"
#include "nnue_test_command.h"

#include <set>
#include <fstream>
#include <random>

#define ASSERT(X) { if (!(X)) { std::cout << "\nError : ASSERT(" << #X << "), " << __FILE__ << "(" << __LINE__ << "): " << __func__ << std::endl; \
 std::this_thread::sleep_for(std::chrono::microseconds(3000)); *(int*)1 =0;} }

namespace Eval {

namespace NNUE {

namespace {

// Testing RawFeatures mainly for difference calculation
void TestFeatures(Position& pos) {
  constexpr std::uint64_t num_games = 1000;
  StateInfo si;
  pos.set(StartFEN, false, &si, Threads.main());

  std::random_device seed_it;
  PRNG prng(seed_it());

  std::uint64_t num_moves = 0;
  std::vector<std::uint64_t> num_updates(kRefreshTriggers.size() + 1);
  std::vector<std::uint64_t> num_resets(kRefreshTriggers.size());
  constexpr IndexType kUnknown = -1;
  std::vector trigger_map(RawFeatures::kDimensions, kUnknown);
  auto make_index_sets = [&](const Position& pos) {
    std::vector index_sets(
        kRefreshTriggers.size(), std::vector<std::set<IndexType>>(2));
    for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
      Features::IndexList active_indices[2];
      RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i],
                                       active_indices);
      for (const auto perspective : Colors) {
        for (const auto index : active_indices[perspective]) {
          ASSERT(index < RawFeatures::kDimensions)
          ASSERT(index_sets[i][perspective].count(index) == 0)
          ASSERT(trigger_map[index] == kUnknown || trigger_map[index] == i)
          index_sets[i][perspective].insert(index);
          trigger_map[index] = i;
        }
      }
    }
    return index_sets;
  };
  auto update_index_sets = [&](const Position& pos, auto* index_sets) {
    for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
      Features::IndexList removed_indices[2], added_indices[2];
      bool reset[2];
      RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[i],
                                        removed_indices, added_indices, reset);
      for (const auto perspective : Colors) {
        if (reset[perspective]) {
          (*index_sets)[i][perspective].clear();
          ++num_resets[i];
        } else {
          for (const auto index : removed_indices[perspective]) {
            ASSERT(index < RawFeatures::kDimensions)
            ASSERT((*index_sets)[i][perspective].count(index) == 1)
            ASSERT(trigger_map[index] == kUnknown || trigger_map[index] == i)
            (*index_sets)[i][perspective].erase(index);
            ++num_updates.back();
            ++num_updates[i];
            trigger_map[index] = i;
          }
        }
        for (const auto index : added_indices[perspective]) {
          ASSERT(index < RawFeatures::kDimensions)
          ASSERT((*index_sets)[i][perspective].count(index) == 0)
          ASSERT(trigger_map[index] == kUnknown || trigger_map[index] == i)
          (*index_sets)[i][perspective].insert(index);
          ++num_updates.back();
          ++num_updates[i];
          trigger_map[index] = i;
        }
      }
    }
  };

  std::cout << "feature set: " << RawFeatures::GetName()
            << "[" << RawFeatures::kDimensions << "]" << std::endl;
  std::cout << "start testing with random games";

  for (std::uint64_t i = 0; i < num_games; ++i) {
	  constexpr int MAX_PLY = 256;
	  auto index_sets = make_index_sets(pos);
    for (int ply = 0; ply < MAX_PLY; ++ply) {
	    StateInfo state[MAX_PLY];
	    MoveList<LEGAL> mg(pos); // Generate all legal hands

      // There was no legal move == Clog
      if (mg.size() == 0)
        break;

      // Randomly choose from the generated moves and advance the phase with the moves.
      const Move m = mg.begin()[prng.rand(mg.size())];
      pos.do_move(m, state[ply]);

      ++num_moves;
      update_index_sets(pos, &index_sets);
	    ASSERT(index_sets == make_index_sets(pos))
    }

    pos.set(StartFEN, false, &si, Threads.main());

    // Output'.' every 100 times (so you can see that it's progressing)
    if (i % 100 == 0)
      std::cout << "." << std::flush;
  }
  std::cout << "passed." << std::endl;
  std::cout << num_games << " games, " << num_moves << " moves, "
            << num_updates.back() << " updates, "
            << 1.0 * static_cast<double>(num_updates.back()) / static_cast<double>(num_moves)
            << " updates per move" << std::endl;
  std::size_t num_observed_indices = 0;
  for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
    const auto count = std::count(trigger_map.begin(), trigger_map.end(), i);
    num_observed_indices += count;
    std::cout << "TriggerEvent(" << static_cast<int>(kRefreshTriggers[i])
              << "): " << count << " features ("
              << 100.0 * static_cast<double>(count) / RawFeatures::kDimensions << "%), "
              << num_updates[i] << " updates ("
              << 1.0 * static_cast<double>(num_updates[i]) / static_cast<double>(num_moves) << " per move), "
              << num_resets[i] << " resets ("
              << 100.0 * static_cast<double>(num_resets[i]) / static_cast<double>(num_moves) << "%)"
              << std::endl;
  }
  std::cout << "observed " << num_observed_indices << " ("
            << 100.0 * num_observed_indices / RawFeatures::kDimensions
            << "% of " << RawFeatures::kDimensions
            << ") features" << std::endl;
}

// Output a string that represents the structure of the evaluation function
void PrintInfo(std::istream& stream) {
  std::cout << "network architecture: " << GetArchitectureString() << std::endl;

  while (true) {
    std::string file_name;
    stream >> file_name;
    if (file_name.empty()) break;

    std::uint32_t hash_value;
    std::string architecture;
    const bool success = [&]
    {
      std::ifstream file_stream(file_name, std::ios::binary);
      if (!file_stream) return false;
      if (!ReadHeader(file_stream, &hash_value, &architecture)) return false;
      return true;
    }();

    std::cout << file_name << ": ";
    if (success) {
      if (hash_value == kHashValue) {
        std::cout << "matches with this binary";
        if (architecture != GetArchitectureString()) {
          std::cout << ", but architecture string differs: " << architecture;
        }
        std::cout << std::endl;
      } else {
        std::cout << architecture << std::endl;
      }
    } else {
      std::cout << "failed to read header" << std::endl;
    }
  }
}

}  // namespace

// USI extended command for NNUE evaluation function
void TestCommand(Position& pos, std::istream& stream) {
  std::string sub_command;
  stream >> sub_command;

  if (sub_command == "test_features") {
    TestFeatures(pos);
  } else if (sub_command == "info") {
    PrintInfo(stream);
  } else {
    std::cout << "usage:" << std::endl;
    std::cout << " test nnue test_features" << std::endl;
    std::cout << " test nnue info [path/to/" << fileName << "...]" << std::endl;
  }
}

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(ENABLE_TEST_CMD) && defined(EVAL_NNUE)
