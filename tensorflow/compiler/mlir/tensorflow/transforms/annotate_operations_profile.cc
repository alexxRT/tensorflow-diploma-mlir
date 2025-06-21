/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {

#define GEN_PASS_DEF_ANNOTATEOPERATIONSPROFILEPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

namespace TF {

struct AnnotateOperationsProfilePass
    : public impl::AnnotateOperationsProfilePassBase<
          AnnotateOperationsProfilePass> {

  AnnotateOperationsProfilePass() = default;
  explicit AnnotateOperationsProfilePass(std::string profilePath)
      : profileDataFilePath(std::move(profilePath)) {}

  void runOnOperation() override;

  private:
    void readProfilerData(ProfilerData* data, Operation* op);
    std::string profileDataFilePath;
};

void AnnotateOperationsProfilePass::runOnOperation() {
  ModuleOp op = getOperation();

  op.walk([&](mlir::Operation* nestedOp) {
    if (nestedOp->getDialect() &&
        nestedOp->getDialect()->getNamespace() == "tf") {
        if (nestedOp->hasTrait<ProfileAnnotation>()) {
            // TODO: readProfilerData();
            ProfilerData data;
            readProfilerData(&data, nestedOp);
            nestedOp->AttachProfilerData(data);
        }
    }
  });
}

void AnnotateOperationsProfilePass::readProfilerData(ProfilerData* data, Operation* op) {
    StringRef opName = op->getName()->getStringRef();
    Location opLoc = op->getLoc(); // might be useful for mapping

    // TODO: Add read profile from file and mapping to current node
    data->timestamp = 0;
    data->duration = 0;
}

std::unique_ptr<OperationPass<ModuleOp>>
CreateAnnotateOperationsProfilerPass(const std::string& profileFilePath) {
  return std::make_unique<AnnotateOperationsProfilePass>(profileFilePath);
}

}  // namespace TF
}  // namespace mlir
