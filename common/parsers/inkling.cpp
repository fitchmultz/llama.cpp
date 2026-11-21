#include "parsers.h"

#include <algorithm>
#include <unordered_map>

bool is_inkling_template(const std::string & src) {
    return src.find("<|message_user|>") != std::string::npos &&
           src.find("<|content_image|>") != std::string::npos;
}

static json inkling_sort_json_keys(const json & value) {
    if (value.is_object()) {
        std::vector<std::string> keys;
        keys.reserve(value.size());
        for (const auto & [k, _] : value.items()) {
            keys.push_back(k);
        }
        std::sort(keys.begin(), keys.end());
        json sorted = json::object();
        for (const auto & k : keys) {
            sorted[k] = inkling_sort_json_keys(value.at(k));
        }
        return sorted;
    }
    if (value.is_array()) {
        json arr = json::array();
        for (const auto & el : value) {
            arr.push_back(inkling_sort_json_keys(el));
        }
        return arr;
    }
    return value;
}

static std::string inkling_format_tool_declare(const json & tools_arr) {
    json specs = json::array();
    for (const auto & tool : tools_arr) {
        const json & fn = (tool.is_object() && tool.contains("function")) ? tool.at("function") : tool;
        if (!fn.is_object() || !fn.contains("name") || !fn.at("name").is_string()) {
            continue;
        }
        const std::string desc = (fn.contains("description") && fn.at("description").is_string())
            ? fn.at("description").get<std::string>()
            : "";
        const json params = (fn.contains("parameters") && !fn.at("parameters").is_null())
            ? inkling_sort_json_keys(fn.at("parameters"))
            : json::object();
        const std::string type = (tool.is_object() && tool.contains("type") && tool.at("type").is_string() && !tool.at("type").get<std::string>().empty())
            ? tool.at("type").get<std::string>()
            : "function";
        json spec = json::object();
        spec["description"] = desc;
        spec["name"]        = fn.at("name");
        spec["parameters"]  = params;
        spec["type"]        = type;
        specs.push_back(std::move(spec));
    }
    return "<|message_system|>tool_declare<|content_xml|>" + specs.dump() + "<|end_message|>";
}

static bool inkling_msg_has_nonempty_content(const json & msg) {
    if (!msg.contains("content") || msg.at("content").is_null()) {
        return false;
    }
    const auto & content = msg.at("content");
    if (content.is_string()) {
        return !content.get<std::string>().empty();
    }
    if (content.is_array()) {
        return !content.empty();
    }
    return true;
}

// Replace sentinel system text blocks emitted for mid-conversation `role: "system", tools: [...]` messages
// with native Inkling `<|message_system|>tool_declare<|content_xml|>...<|end_message|>` template parts (`is_input = false`).
static void inkling_expand_mid_convo_tools(
        common_chat_prompt & prompt,
        const std::unordered_map<std::string, std::string> & decls) {
    if (decls.empty()) {
        return;
    }
    const std::string OPEN = "<|message_system|><|content_text|>";
    const std::string END  = "<|end_message|>";

    std::vector<common_chat_prompt_part> out;
    out.reserve(prompt.parts.size());
    for (size_t i = 0; i < prompt.parts.size(); ++i) {
        const auto & part = prompt.parts[i];
        auto it = part.is_input ? decls.find(part.text) : decls.end();
        if (it != decls.end() &&
            !out.empty() && !out.back().is_input && string_ends_with(out.back().text, OPEN) &&
            i + 1 < prompt.parts.size() && !prompt.parts[i + 1].is_input && string_starts_with(prompt.parts[i + 1].text, END)) {
            out.back().text.resize(out.back().text.size() - OPEN.size());
            out.back().text += it->second;
            const std::string tail = prompt.parts[i + 1].text.substr(END.size());
            if (!tail.empty()) {
                out.back().text += tail;
            }
            ++i; // consumed the closing <|end_message|> part
            continue;
        }
        if (!out.empty() && out.back().is_input == part.is_input) {
            out.back().text += part.text;
        } else {
            out.push_back(part);
        }
    }
    prompt.parts = std::move(out);
}

// the template makes one message block per content part, but a media marker comes inside the text of a message
// <header><|content_text|>{media}text<|end_message|> --> <header>{media}<|end_message|><header><|content_text|>text<|end_message|>
static void inkling_split_media(common_chat_prompt & prompt) {
    const std::string TEXT      = "<|content_text|>";
    const std::string END       = "<|end_message|>";
    const std::string MEDIA_BEG = "<__media_";
    const std::string MEDIA_END = "__>";

    std::vector<common_chat_prompt_part> out;
    for (size_t i = 0; i < prompt.parts.size(); i++) {
        const auto & part = prompt.parts[i];
        const bool in_text_block = part.is_input && part.text.find(MEDIA_BEG) != std::string::npos &&
            !out.empty() && !out.back().is_input && string_ends_with(out.back().text, TEXT) &&
            i + 1 < prompt.parts.size() && !prompt.parts[i + 1].is_input && string_starts_with(prompt.parts[i + 1].text, END);
        const size_t header_pos = in_text_block ? out.back().text.rfind("<|message_") : std::string::npos;
        if (header_pos == std::string::npos) {
            out.push_back(part);
            continue;
        }
        const std::string header = out.back().text.substr(header_pos, out.back().text.size() - TEXT.size() - header_pos);

        std::string rest = part.text;
        for (bool first = true; !rest.empty(); first = false) {
            const size_t beg      = rest.find(MEDIA_BEG);
            const size_t end      = beg == std::string::npos ? beg : rest.find(MEDIA_END, beg);
            const bool   is_media = beg == 0 && end != std::string::npos;
            const size_t len      = is_media ? end + MEDIA_END.size() : (end == std::string::npos ? rest.size() : beg);
            if (!first) {
                out.push_back({ END + header + (is_media ? "" : TEXT), false });
            } else if (is_media) {
                // mtmd opens the media block
                out.back().text.resize(out.back().text.size() - TEXT.size());
            }
            out.push_back({ rest.substr(0, len), true });
            rest.erase(0, len);
        }
    }
    prompt.parts = std::move(out);
}

// Inkling 1 and 2 - the assistant turn is a list of blocks, all closed by <|end_message|>:
//   <|message_model|><|content_thinking|>...<|end_message|>
//   <|message_model|><|content_text|>...<|end_message|>
//   <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
common_chat_params common_chat_params_init_inkling(const common_chat_template &          tmpl,
                                                   const autoparser::generation_params & inputs) {
    common_chat_params data;

    json effective_tools = json::array();
    std::unordered_map<std::string, size_t> tool_index_by_name;
    auto merge_tools = [&](const json & tools_arr) {
        if (!tools_arr.is_array()) {
            return;
        }
        for (const auto & tool : tools_arr) {
            if (!tool.is_object()) {
                continue;
            }
            json normalized = tool;
            if (!normalized.contains("function")) {
                normalized = json{
                    { "type",     "function" },
                    { "function", tool },
                };
            }
            const auto & fn = normalized.at("function");
            if (!fn.is_object() || !fn.contains("name") || !fn.at("name").is_string()) {
                continue;
            }
            const std::string name = fn.at("name").get<std::string>();
            auto it = tool_index_by_name.find(name);
            if (it != tool_index_by_name.end()) {
                effective_tools[it->second] = std::move(normalized);
            } else {
                tool_index_by_name[name] = effective_tools.size();
                effective_tools.push_back(std::move(normalized));
            }
        }
    };

    merge_tools(inputs.tools);

    bool has_mid_convo_tools = false;
    if (inputs.messages.is_array()) {
        for (const auto & msg : inputs.messages) {
            if (msg.is_object() &&
                msg.contains("role") && msg.at("role") == "system" &&
                msg.contains("tools") && msg.at("tools").is_array() && !msg.at("tools").empty()) {
                has_mid_convo_tools = true;
                merge_tools(msg.at("tools"));
            }
        }
    }

    std::unordered_map<std::string, std::string> mid_tool_decls;
    std::optional<json> messages_override;
    if (has_mid_convo_tools) {
        const std::string raw_msgs = inputs.messages.dump();
        std::string sentinel_prefix = "\x1e__inkling_tool_declare_";
        while (raw_msgs.find(sentinel_prefix) != std::string::npos) {
            sentinel_prefix += "_";
        }

        json rewritten = json::array();
        size_t decl_idx = 0;
        for (const auto & msg : inputs.messages) {
            if (msg.is_object() &&
                msg.contains("role") && msg.at("role") == "system" &&
                msg.contains("tools") && msg.at("tools").is_array() && !msg.at("tools").empty()) {
                const std::string sentinel = sentinel_prefix + std::to_string(decl_idx++) + "__\x1e";
                mid_tool_decls[sentinel] = inkling_format_tool_declare(msg.at("tools"));
                rewritten.push_back(json{
                    { "role",    "system" },
                    { "content", sentinel },
                });
                if (inkling_msg_has_nonempty_content(msg)) {
                    json rest = msg;
                    rest.erase("tools");
                    rewritten.push_back(std::move(rest));
                }
            } else {
                rewritten.push_back(msg);
            }
        }
        messages_override = std::move(rewritten);
    }

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs, messages_override);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs, messages_override);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;

    inkling_expand_mid_convo_tools(data.prompt, mid_tool_decls);
    inkling_split_media(data.prompt);

    const std::string HEADER    = "<|message_model|>";
    const std::string THINK     = "<|content_thinking|>";
    const std::string TEXT      = "<|content_text|>";
    const std::string TOOL_JSON = "<|content_invoke_tool_json|>";
    const std::string END       = "<|end_message|>";

    data.preserved_tokens = { HEADER, THINK, TEXT, TOOL_JSON, END };

    data.thinking_start_tag = THINK;
    data.thinking_end_tags  = { END };

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_ASSISTANT, HEADER               },
        { COMMON_CHAT_ROLE_USER,      "<|message_user|>"   },
        { COMMON_CHAT_ROLE_SYSTEM,    "<|message_system|>" },
        { COMMON_CHAT_ROLE_TOOL,      "<|message_tool|>"   },
    };

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt.clear();
        if (!msg.reasoning_content.empty()) {
            data.generation_prompt = HEADER + THINK + msg.reasoning_content;
        }
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            if (!data.generation_prompt.empty()) {
                data.generation_prompt += END;
            }
            data.generation_prompt += HEADER + TEXT + msg.render_content();
        }

        data.prompt.append(data.generation_prompt);
    }

    auto has_tools           = effective_tools.is_array() && !effective_tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto use_tools           = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto header = p.literal(HEADER);
        auto end    = p.literal(END);

        auto think    = p.literal(THINK);
        auto thinking = p.rule("thinking", extract_reasoning ? think + p.reasoning(p.until(END)) + end :
                                                               p.atomic(p.content(think)) + p.content(p.until(END) + end));
        auto text     = p.rule("text", p.literal(TEXT) + p.content(p.until(END)) + end);

        if (has_response_format) {
            auto response_format = p.rule("response-format", p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)));
            return p.zero_or_more(header + thinking) + header + p.literal(TEXT) + response_format + end;
        }

        auto blocks = p.zero_or_more(header + (thinking | text));

        if (!use_tools) {
            return blocks;
        }

        auto tool_choice = p.choice();
        foreach_function(effective_tools, [&](size_t tool_index, const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");

            auto open = p.tool_name(p.literal(name)) + p.literal(TOOL_JSON) + p.literal("{\"name\":" + json(name).dump() + ",\"args\":");
            auto args = p.tool_args(p.json());

            tool_choice |= p.rule("tool-" + std::to_string(tool_index), p.tool(p.tool_open(open) + args + p.tool_close(p.literal("}")) + end));
        });

        // tool calls are the last blocks of the turn
        auto calls      = inputs.parallel_tool_calls ? tool_choice + p.zero_or_more(header + tool_choice) : tool_choice;
        auto tool_calls = header + p.trigger_rule("tool-calls", calls);

        if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
            return blocks + tool_calls;
        }

        return blocks + p.optional(tool_calls);
    });

    if (has_response_format || use_tools) {
        data.grammar_lazy = !has_response_format && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, data.grammar_lazy);
        });

        if (data.grammar_lazy) {
            // a header followed by text and not by a content token is a tool call, the grammar starts at the name
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "<\\|message_model\\|>([^<])" },
            };
            // Inkling 1: the first header is in the generation prompt, and a lazy grammar does not get the prompt.
            // an output that starts with no content token is a tool call. not for a continuation, it starts with text
            if (!inputs.has_continuation()) {
                data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL, "[^<]+" });
            }
        }
    }

    return data;
}
