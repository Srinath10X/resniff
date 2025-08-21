#include <bits/stdc++.h>
#include <cpr/cpr.h>
#include <crow/crow.h>
#include <mutex>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using request = crow::request;
using response = crow::response;
using HTTP_Method = crow::HTTPMethod;

typedef struct {
  std::string prompt;
  std::vector<double> embedding;
} VectorEntry;

std::mutex VectorDatabaseMutex;
std::vector<VectorEntry> VectorDatabase;

const double cosine_similarity(
    /* clang-format off */
	const std::vector<double> &a,
	const std::vector<double> &b
    /* clang-format on */
) {
  if (a.size() != b.size()) {
    std::cerr
        << "Error: Vectors must be of the same size for cosine similarity."
        << std::endl;
    return 0.0;
  }
  double dot = 0.0, normA = 0.0, normB = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA == 0.0 || normB == 0.0) {
    return 0.0;
  }
  return dot / (std::sqrt(normA) * std::sqrt(normB));
}

std::vector<double> get_embedding(const std::string &prompt) {
  json payload = {{"model", "nomic-embed-text"}, {"prompt", prompt}};

  /* clang-format off */
  cpr::Response res = cpr::Post(
		cpr::Url{"http://localhost:11434/api/embeddings"},
		cpr::Body{payload.dump()},
		cpr::Header{{"Content-Type", "application/json"}}
	);
  /* clang-format on */

  if (res.status_code == 200) {
    try {
      auto response_json = json::parse(res.text);
      std::cout << "Successfully received embedding!" << std::endl;
      return response_json["embedding"].get<std::vector<double>>();
    } catch (const json::parse_error &e) {
      std::cerr << "JSON parse error: " << e.what() << std::endl;
      return {};
    }
  } else {
    std::cerr << "Error getting embedding: " << res.status_code << std::endl;
    std::cerr << "Response Body: " << res.text << std::endl;
    return {};
  }
}

const std::string ask_llm(const std::string &llm_model,
                          const std::string &ctx_prompt) {
  const json payload = {
      {"model", llm_model}, {"prompt", ctx_prompt}, {"stream", false}};

  /* clang-format off */
	cpr::Response res = cpr::Post(
		cpr::Url{"http://localhost:11434/api/generate"},
		cpr::Body{payload.dump()},
		cpr::Header{{"Content-Type", "application/json"}}
	);
  /* clang-format on */

  if (res.status_code == 200) {
    try {
      auto response_json = json::parse(res.text);
      std::cout << "Successfully received response from LLM!" << std::endl;
      return response_json["response"].get<std::string>();
    } catch (const json::parse_error &e) {
      std::cerr << "JSON parse error in ask_llm: " << e.what() << std::endl;
      return "Error parsing LLM response.";
    }
  } else {
    std::cerr << "Error asking LLM: " << res.status_code << std::endl;
    std::cerr << "Response Body: " << res.text << std::endl;
    return "Error communicating with LLM.";
  }
}

int main() {
  crow::SimpleApp App;

  CROW_ROUTE(App, "/api/ping")([]() { return "pong!"; });

  CROW_ROUTE(App, "/api/chat")
      .methods(HTTP_Method::POST)([](const request &req) {
        const auto body = crow::json::load(req.body);

        if (!body) {
          return crow::response(400, "Invalid JSON");
        }

        const std::string prompt = body["prompt"].s();
        const std::string llm_model = body["model"].s();

        if (prompt.empty() || llm_model.empty()) {
          return crow::response(400, "Missing 'prompt' or 'model' field.");
        }

        std::vector<double> prompt_vector_embedding = get_embedding(prompt);

        if (prompt_vector_embedding.empty()) {
          return crow::response(500, "Failed to get embedding for the prompt.");
        }

        double highest_similarity = -1.0; // Use double for consistency
        std::string most_relevant_context = "No relevant context found.";

        std::lock_guard<std::mutex> lock(VectorDatabaseMutex);

        if (!VectorDatabase.empty()) {
          for (const auto &entry : VectorDatabase) {
            double sim =
                cosine_similarity(prompt_vector_embedding, entry.embedding);
            if (sim > highest_similarity) {
              highest_similarity = sim;
              most_relevant_context = entry.prompt;
            }
          }
        }

        std::cout << "Highest similarity: " << highest_similarity << std::endl;
        std::cout << "Most relevant context: " << most_relevant_context
                  << std::endl;

        VectorDatabase.push_back({prompt, prompt_vector_embedding});

        std::string final_prompt =
            "Based on the following context, please answer the question.\n\n";

        final_prompt += "Context:\n---\n" + most_relevant_context + "\n---\n\n";
        final_prompt += "Question: " + prompt + "\n\nAnswer:";

        const std::string answer = ask_llm(llm_model, final_prompt);

        crow::json::wvalue json_answer = crow::json::wvalue(
            {{"answer", answer}, {"context", most_relevant_context}});

        return crow::response(json_answer);
      });

  App.port(3000).multithreaded().run();

  return 0;
}
