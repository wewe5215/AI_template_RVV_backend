//  Copyright (c) Meta Platforms, Inc. and affiliates.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
#pragma once
// Some helpful unique_ptr instantiations and factory functions for CUDA types
#include <functional>
#include <memory>
#include <type_traits>

#include "device_functions-generated.h"
#include "macros.h"

namespace ait {

// RAII wrapper for owned memory. Not that the underlying calls
// to malloc/free are synchronous for simplicity.
using Ptr = std::unique_ptr<void, std::function<void(void*)>>;

inline Ptr RAII_DeviceMalloc(
    size_t num_bytes) {
  auto* output = malloc(num_bytes);
  auto deleter = [](void* ptr) { free(ptr); };
  return Ptr(output, deleter);
}


} // namespace ait
