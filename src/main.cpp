#include <bits/stdc++.h>
#include <cpr/cpr.h>
#include <crow/crow.h>
#include <mutex>
#include <nlohmann/json.hpp>

/// Do I really need these much of them?
using json = nlohmann::json;
using request = crow::request;
using response = crow::response;
using HTTP_Method = crow::HTTPMethod;

typedef struct {
  std::string prompt;
  std::vector<double> embedding;
} VectorEntry;

/// This mutex if to avoid race condition 🗣️
std::mutex VectorDatabaseMutex;
std::vector<VectorEntry> VectorDatabase;

const double cosine_similarity(
    /* clang-format off */
	const std::vector<double> &a, 
	const std::vector<double> &b
    /* clang-format on */
) {
  double dot = 0.0, normA = 0.0, normB = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
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
    auto response_json = json::parse(res.text);
    std::cout << "Successfully received embedding!" << std::endl;
    return response_json["embedding"].get<std::vector<double>>();
  } else {
    std::cerr << "Error: " << res.status_code << std::endl;
    std::cerr << "Response Body: " << res.text << std::endl;
    return {};
  }
}

const std::string ask_llm(const std::string &llm_model,
                          const std::string &ctx_prompt) {
  const json payload = {{"model", llm_model}, {"prompt", ctx_prompt}};

  /* clang-format off */
	cpr::Response res = cpr::Post(
		cpr::Url{"http://localhost:11434/api/completions"}, 
		cpr::Body{payload.dump()}, 
		cpr::Header{{"Content-Type", "application/json"}}
	);
  /* clang-format on */
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

        if (prompt.empty()) {
          return crow::response(400, "Missing prompt");
        }

        std::vector<double> prompt_vector_embedding = get_embedding(prompt);

        if (prompt_vector_embedding.empty()) {
          return crow::response(500, "Failed to get embedding");
        }

        std::lock_guard<std::mutex> lock(VectorDatabaseMutex);
        VectorDatabase.push_back({prompt, prompt_vector_embedding});

        // for (const auto &entry : VectorDatabase) {
        // 	std::cout << entry.prompt << std::endl;
        //
        //   for (size_t idx = 0; idx < entry.embedding.size(); idx++) {
        // 	  std::cout << entry.embedding[idx] << " ";
        //   }
        // }

        float highest_similarity = -1.0f;
        std::string most_relevant_context = "No relevant context found.";

        for (const auto &entry : VectorDatabase) {
          double sim = cosine_similarity(entry.embedding, entry.embedding);
          if (sim > highest_similarity) {
            highest_similarity = sim;
            most_relevant_context = entry.prompt;
          }
        }

        std::cout << "Highest similarity: " << highest_similarity << std::endl;
        std::cout << "Most relevant context: " << most_relevant_context
                  << std::endl;

        std::string final_prompt =
            "Based on the following context, please answer the question.\n\n";

        final_prompt += "Context:\n---\n" + most_relevant_context + "\n---\n\n";
        final_prompt += "Question: " + prompt + "\n\nAnswer:";

        std::string answer = ask_llm();

        return crow::response(200);
      });

  App.port(3000).multithreaded().run();

  return 0;
}
