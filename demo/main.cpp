#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include <cxxopts.hpp>
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
  cxxopts::Options options("LLaMA Text Generation", "Generate text using LLaMA model");

  options.add_options()("m,model", "Path to model checkpoint (e.g., out/model.bin)",
                        cxxopts::value<std::string>())("t,tokenizer", "Path to tokenizer",
                                                       cxxopts::value<std::string>())(
      "d,device", "Device to use: cpu, cuda", cxxopts::value<std::string>()->default_value("cuda"))(
      "s,steps", "Number of generation steps", cxxopts::value<int>()->default_value("128"))(
      "o,output", "Print generated text", cxxopts::value<bool>()->default_value("true"))(
      "q,quant", "Use quantized model", cxxopts::value<bool>()->default_value("false"))(
      "h,help", "Print usage");

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  if (!result.count("model") || !result.count("tokenizer")) {
    std::cerr << "Error: Missing required arguments: --model and --tokenizer are required."
              << std::endl;
    std::cerr << options.help() << std::endl;
    return 1;
  }

  std::string checkpoint_path = result["model"].as<std::string>();
  std::string tokenizer_path = result["tokenizer"].as<std::string>();
  std::string device_str = result["device"].as<std::string>();
  int total_steps = result["steps"].as<int>();
  bool need_output = result["output"].as<bool>();
  bool is_quant_model = result["quant"].as<bool>();

  // 设备映射
  base::DeviceType device_type = base::DeviceType::kDeviceUnknown;
  if (device_str == "cpu") {
    device_type = base::DeviceType::kDeviceCPU;
  } else if (device_str == "cuda") {
    device_type = base::DeviceType::kDeviceCUDA;
  } else {
    std::cerr << "Error: Invalid device type: " << device_str << std::endl;
    std::cerr << options.help() << std::endl;
    return 1;
  }

  model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path, checkpoint_path,
                           is_quant_model);

  base::DeviceType base_device_type = (device_type == base::DeviceType::kDeviceCUDA)
                                          ? base::DeviceType::kDeviceCUDA
                                          : base::DeviceType::kDeviceCPU;
  auto init_status = model.init(base_device_type);
  if (!init_status) {
    std::cerr << "Error: Model initialization failed: " << init_status.get_err_msg() << std::endl;
    return 1;
  }

  std::string sentence = "hello";
  auto start = std::chrono::steady_clock::now();
  std::cout << "Generating..." << std::endl;
  int steps = generate(model, sentence, total_steps, need_output);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration<double>(end - start).count();
  std::cout << "\nsteps/s: " << (static_cast<double>(steps) / duration) << std::endl;

  return 0;
}