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
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <unordered_map>

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
#include "llvm/Support/Casting.h" // from @llvm-project
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
      : profileDataFilePath(std::move(profilePath)) {

    ModuleOp op = getOperation();
    std::ifstream file(profileDataFilePath);

    if (!file) {
      llvm::outs() << "[TFProfileGuided] Error! Unable to read from file " << profileDataFilePath << "\n" << "All times will be set to 0!\n";
      op.emitError("Failed to read profile from input file!");
      return;
    }

    std::string line;
    std::getline(file, line); // read first line to check columns names

    std::vector<std::string> fileHeader = parseByDelimeter(line, ',');
    if (fileHeader.size() == 0 ||
        fileHeader[0] != std::string("name") ||
        fileHeader[1] != std::string("ts") ||
        fileHeader[2] != std::string("duration")) {
        llvm::outs() << "[TFProfileGuided] Error! Wrong header format in file: " << profileDataFilePath << "\n" << "All times will be set to 0!\n";
        op.emitError("Failed to read profile from input file!");
        return;
    }

    while (std::getline(file, line)) { // reading profile data to later use on operations
      std::vector<std::string> opData = parseByDelimeter(line, ',');
      readData.insert({opData[0], ProfilerData(std::stol(opData[1]), std::stol(opData[2]))});
    }

    file.close();
  };

  void runOnOperation() override;

  private:
    std::vector<std::string> parseByDelimeter(std::string& str, const char delimiter);
    std::unordered_map<std::string, ProfilerData> readData;
    std::string profileDataFilePath;
};

void AnnotateOperationsProfilePass::runOnOperation() {
  ModuleOp op = getOperation();

  op.walk([&](mlir::Operation* nestedOp) {
    if (nestedOp->hasTrait<mlir::OpTrait::TF::ProfileAnnotation>()) {
        ProfilerData data(0, 0);

        std::string opName = nestedOp->getName().getStringRef().str();
        if (readData.find(opName) != readData.end())
          data = readData.at(opName);

        auto profileInterface = dyn_cast<ProfileAnnotationInterface>(nestedOp);
        profileInterface.AttachProfilerData(data);
    }
  });
}

std::vector<std::string> AnnotateOperationsProfilePass::parseByDelimeter(std::string& str, const char delimeter) {
 std::vector<std::string> parsed;

  if (str.empty())
    return parsed;

  std::stringstream ss(str);
  std::string entry;

  while (std::getline(ss, entry, delimeter)) {
    parsed.push_back(entry);
  }
  return parsed;
}

std::unique_ptr<OperationPass<ModuleOp>>
CreateAnnotateOperationsProfilePass(const std::string& profileFilePath) {
  return std::make_unique<AnnotateOperationsProfilePass>(profileFilePath);
}

std::unique_ptr<OperationPass<ModuleOp>>
CreateAnnotateOperationsProfilePass() {
  return std::make_unique<AnnotateOperationsProfilePass>();
}

}  // namespace TF
}  // namespace mlir
