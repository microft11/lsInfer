#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include <CLI/CLI.hpp>
#include "model/llama3.h"
int32_t generate(const model::LLama2Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
  auto tokens = model.encode(sentence);
  int32_t prompt_len = tokens.size();
  LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

  int32_t pos = 0;
  int32_t next = -1;
  bool is_prompt = true;
  const auto& prompt_embedding = model.embedding(tokens);
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  std::vector<int32_t> words;
  while (pos < total_steps) {
    pos_tensor.index<int32_t>(0) = pos;
    if (pos < prompt_len - 1) {
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    } else {
      is_prompt = false;
      tokens = std::vector<int32_t>{next};
      const auto& token_embedding = model.embedding(tokens);
      tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
      model.predict(input, pos_tensor, is_prompt, next);
    }
    if (model.is_sentence_ending(next)) {
      break;
    }
    if (is_prompt) {
      next = tokens.at(pos + 1);
      words.push_back(next);
    } else {
      words.push_back(next);
    }

    pos += 1;
  }
  if (need_output) {
    printf("%s ", model.decode(words).data());
    fflush(stdout);
  }
  return std::min(pos, total_steps);
}

int main(int argc, char* argv[]) {
  CLI::App app{"LLaMA Text Generation"};

  std::string checkpoint_path;
  std::string tokenizer_path;
  std::string device_str = "cpu";  // 默认设备
  int total_steps = 128;
  bool need_output = false;

  // 添加命令行参数
  app.add_option("-m,--model", checkpoint_path, "Path to model checkpoint (e.g., out/model.bin)")
      ->required();
  app.add_option("-t,--tokenizer", tokenizer_path, "Path to tokenizer")->required();
  app.add_option("-d,--device", device_str, "Device to use: cpu, cuda, rocm (default: cuda)");
  app.add_option("-s,--steps", total_steps, "Number of generation steps (default: 128)");
  app.add_flag("-o,--output", need_output, "Print generated text");

  CLI11_PARSE(app, argc, argv);

  // 设备映射
  base::DeviceType device_type = base::DeviceType::kDeviceUnknown;
  if (device_str == "cpu") {
    device_type = base::DeviceType::kDeviceCPU;
  } else if (device_str == "cuda") {
    device_type = base::DeviceType::kDeviceCUDA;
  } else {
    LOG(FATAL) << "Invalid device type: " << device_str;
  }

  model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path, checkpoint_path, false);

  // 选择设备初始化
  base::DeviceType base_device_type = base::DeviceType::kDeviceCPU;
  if (device_type == base::DeviceType::kDeviceCUDA) {
    base_device_type = base::DeviceType::kDeviceCUDA;
  }

  auto init_status = model.init(base_device_type);
  if (!init_status) {
    LOG(FATAL) << "Model initialization failed: " << init_status.get_err_msg();
  }

  const std::string& sentence = "a";
  auto start = std::chrono::steady_clock::now();
  printf("Generating...\n");
  fflush(stdout);
  int steps = generate(model, sentence, total_steps, need_output);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
  fflush(stdout);

  return 0;
}
