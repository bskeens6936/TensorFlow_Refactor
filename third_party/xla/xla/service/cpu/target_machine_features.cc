/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/cpu/target_machine_features.h"

#include <algorithm>

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/MathExtras.h"
#include "xla/cpu_function_runtime.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace cpu {

llvm::TargetTransformInfo* LLVMTargetMachineFeatures::GetTargetTransformInfoFor(
    const llvm::Function& function) const {
  auto it = target_transform_info_cache_.find(&function);
  if (it == target_transform_info_cache_.end()) {
    auto emplace_result = target_transform_info_cache_.emplace(
        &function, target_machine_->getTargetTransformInfo(function));
    CHECK(emplace_result.second);
    it = emplace_result.first;
  }

  return &it->second;
}

int64_t LLVMTargetMachineFeatures::minimum_alignment_for_allocation(
    int64_t size_bytes) const {
  // Assume that all pointers are aligned to at least
  // xla::cpu_function_runtime::kMinAlign.
  if (size_bytes == 0) {
    // No need to align empty buffers.
    return 1;
  }

  // Allow small buffers to be underaligned, there is no vectorization benefit
  // anyways.
  return std::min<int64_t>(llvm::PowerOf2Ceil(size_bytes),
                           cpu_function_runtime::MinAlign());
}

std::string LLVMTargetMachineFeatures::get_target_feature_string() const {
  return target_machine_->getTargetFeatureString().str();
}

}  // namespace cpu
}  // namespace xla
