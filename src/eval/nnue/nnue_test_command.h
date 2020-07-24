// USI extended command interface for NNUE evaluation function

#pragma once

#if defined(ENABLE_TEST_CMD) && defined(EVAL_NNUE)

namespace Eval {

	namespace NNUE {

		// USI extended command for NNUE evaluation function
		void TestCommand(Position& pos, std::istream& stream);

	}  // namespace NNUE

}  // namespace Eval

#endif  // defined(ENABLE_TEST_CMD) && defined(EVAL_NNUE)

