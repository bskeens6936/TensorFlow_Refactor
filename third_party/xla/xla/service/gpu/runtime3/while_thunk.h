/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_SERVICE_GPU_RUNTIME3_WHILE_THUNK_H_
#define XLA_SERVICE_GPU_RUNTIME3_WHILE_THUNK_H_

#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/runtime3/sequential_thunk.h"
#include "xla/service/gpu/thunk.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// WhileThunk implements the while instruction on GPU by invoking a thunk
// sequence for the while 'condition' computation, and (conditionally) another
// thunk sequence for the while 'body' computation. WhileThunk assumes that
// buffers for the following set of while-related instructions share the same
// allocation:
//   init, condition.parameter, body.parameter, body.root, while.result
// WhileThunk synchronizes the stream to test the result of the 'condition'
// computation.
class WhileThunk : public Thunk {
 public:
  // Constructs a WhileThunk to compute while instruction 'hlo'.
  WhileThunk(ThunkInfo thunk_info,
             const BufferAllocation::Slice& condition_result_buffer_index,
             std::unique_ptr<ThunkSequence> condition_thunk_sequence,
             std::unique_ptr<ThunkSequence> body_thunk_sequence);
  WhileThunk(const WhileThunk&) = delete;
  WhileThunk& operator=(const WhileThunk&) = delete;

  Status Initialize(se::StreamExecutor* executor,
                    ExecutableSource src) override;
  Status ExecuteOnStream(const ExecuteParams& params) override;

  SequentialThunk* condition_thunk_sequence() const {
    return condition_thunk_sequence_.get();
  }
  SequentialThunk* body_thunk_sequence() const {
    return body_thunk_sequence_.get();
  }

  const BufferAllocation::Slice& condition_result_buffer() const {
    return condition_result_buffer_index_;
  }

 private:
  const BufferAllocation::Slice condition_result_buffer_index_;
  std::unique_ptr<SequentialThunk> condition_thunk_sequence_;
  std::unique_ptr<SequentialThunk> body_thunk_sequence_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME3_WHILE_THUNK_H_
